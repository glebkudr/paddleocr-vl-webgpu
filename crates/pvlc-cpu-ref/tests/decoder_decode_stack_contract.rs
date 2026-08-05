use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::UnsafeCell,
    mem::size_of,
};

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderLayerConfig, DecoderLayerDecodeTrace,
    DecoderLayerParameters, DecoderPrefillKvCache, DecoderStackCheckpoint, DecoderStackConfig,
    DecoderStackDecodeTrace, decoder_layer_decode_f32, decoder_layer_prefill_f32,
    decoder_stack_decode_f32, pinned_decoder_stack_decode_f32, rms_norm_f32,
};

// Compact M6b decode-stack contract only. Logits, top-k, generation, and
// random multi-step cache split/reuse flows remain later M6 milestones.

const HIDDEN: usize = 5;
const INTERMEDIATE: usize = 7;
const QUERY_HEADS: usize = 4;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 6;
const EPSILON: f32 = 1.0e-5;
const SECTIONS: [usize; 3] = [1, 1, 1];
const LAYERS: usize = 4;
const CHECKPOINTS: [usize; 3] = [0, 2, 3];
const PREFIX_BOUNDARIES: [usize; 8] = [1, 2, 15, 16, 17, 31, 32, 33];

const PINNED_LAYERS: usize = 18;
const PINNED_HIDDEN: usize = 1_024;
const PINNED_INTERMEDIATE: usize = 3_072;
const PINNED_QUERY_HEADS: usize = 16;
const PINNED_KEY_VALUE_HEADS: usize = 2;
const PINNED_HEAD_DIM: usize = 128;
const PINNED_EPSILON: f32 = 1.0e-5;
const PINNED_SECTIONS: [usize; 3] = [16, 24, 24];

const TRACKED_ALLOCATION_SLOTS: usize = 4_096;
const LIVENESS_MARKER_FLOATS: usize = 16_384;
const LIVENESS_MARKER_VECTORS: usize = 4;
const LIVENESS_MARKER_BYTES: usize =
    LIVENESS_MARKER_FLOATS * LIVENESS_MARKER_VECTORS * size_of::<f32>();
const LIVENESS_ACCOUNTING_SLACK: usize = 16 * 1_024;

// Derived independently from the manual oracle in this file. It covers only
// the compact decode-stack chain, not later M6 logits/top-k/generation paths.
const COMPACT_DECODE_STACK_BLAKE3: &str =
    "be9b12da8ae25bea13ee9469e2b957723c1dbb9b654e8e2f024768e0b4a0070b";

struct ThreadCountingAllocator;

struct ThreadAllocationState {
    enabled: bool,
    live_bytes: usize,
    peak_bytes: usize,
    overflowed: bool,
    pointers: [usize; TRACKED_ALLOCATION_SLOTS],
    sizes: [usize; TRACKED_ALLOCATION_SLOTS],
}

impl ThreadAllocationState {
    const fn new() -> Self {
        Self {
            enabled: false,
            live_bytes: 0,
            peak_bytes: 0,
            overflowed: false,
            pointers: [0; TRACKED_ALLOCATION_SLOTS],
            sizes: [0; TRACKED_ALLOCATION_SLOTS],
        }
    }

    fn begin(&mut self) {
        assert!(
            !self.enabled,
            "allocation tracking window is already active"
        );
        self.live_bytes = 0;
        self.peak_bytes = 0;
        self.overflowed = false;
        self.pointers.fill(0);
        self.sizes.fill(0);
        self.enabled = true;
    }

    fn record_allocation(&mut self, pointer: usize, size: usize) {
        if !self.enabled || size == 0 {
            return;
        }
        let Some(slot) = self.pointers.iter().position(|stored| *stored == 0) else {
            self.overflowed = true;
            return;
        };
        self.pointers[slot] = pointer;
        self.sizes[slot] = size;
        let Some(live_bytes) = self.live_bytes.checked_add(size) else {
            self.overflowed = true;
            self.pointers[slot] = 0;
            self.sizes[slot] = 0;
            return;
        };
        self.live_bytes = live_bytes;
        self.peak_bytes = self.peak_bytes.max(live_bytes);
    }

    fn record_deallocation(&mut self, pointer: usize) {
        if !self.enabled {
            return;
        }
        let Some(slot) = self.pointers.iter().position(|stored| *stored == pointer) else {
            return;
        };
        self.live_bytes = self
            .live_bytes
            .checked_sub(self.sizes[slot])
            .expect("tracked allocation accounting underflowed");
        self.pointers[slot] = 0;
        self.sizes[slot] = 0;
    }

    fn finish(&mut self) -> AllocationSnapshot {
        assert!(self.enabled, "allocation tracking window is not active");
        let snapshot = AllocationSnapshot {
            live_bytes: self.live_bytes,
            peak_bytes: self.peak_bytes,
            overflowed: self.overflowed,
        };
        self.enabled = false;
        self.pointers.fill(0);
        self.sizes.fill(0);
        snapshot
    }
}

#[derive(Clone, Copy, Debug)]
struct AllocationSnapshot {
    live_bytes: usize,
    peak_bytes: usize,
    overflowed: bool,
}

thread_local! {
    static THREAD_ALLOCATION_STATE: UnsafeCell<ThreadAllocationState> =
        const { UnsafeCell::new(ThreadAllocationState::new()) };
}

fn with_allocation_state<T>(operation: impl FnOnce(&mut ThreadAllocationState) -> T) -> T {
    THREAD_ALLOCATION_STATE.with(|state| {
        // SAFETY: the state is thread-local and the helper is synchronous.
        operation(unsafe { &mut *state.get() })
    })
}

fn try_with_allocation_state(operation: impl FnOnce(&mut ThreadAllocationState)) {
    let _ = THREAD_ALLOCATION_STATE.try_with(|state| {
        // SAFETY: as above; `try_with` also avoids TLS destructor panics.
        operation(unsafe { &mut *state.get() });
    });
}

unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: exact layout forwarded to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            try_with_allocation_state(|state| {
                state.record_allocation(pointer as usize, layout.size());
            });
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        try_with_allocation_state(|state| state.record_deallocation(pointer as usize));
        // SAFETY: `pointer` and `layout` match the original allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

fn begin_allocation_tracking() {
    with_allocation_state(ThreadAllocationState::begin);
}

fn allocation_snapshot() -> AllocationSnapshot {
    with_allocation_state(|state| AllocationSnapshot {
        live_bytes: state.live_bytes,
        peak_bytes: state.peak_bytes,
        overflowed: state.overflowed,
    })
}

fn finish_allocation_tracking() -> AllocationSnapshot {
    with_allocation_state(ThreadAllocationState::finish)
}

#[derive(Clone, Debug, PartialEq)]
struct OwnedParameters {
    input_norm_weight: Vec<f32>,
    query_weight: Vec<f32>,
    key_weight: Vec<f32>,
    value_weight: Vec<f32>,
    attention_output_weight: Vec<f32>,
    post_attention_norm_weight: Vec<f32>,
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
}

impl OwnedParameters {
    fn borrowed(&self) -> DecoderLayerParameters<'_> {
        DecoderLayerParameters {
            input_norm_weight: &self.input_norm_weight,
            query_weight: &self.query_weight,
            key_weight: &self.key_weight,
            value_weight: &self.value_weight,
            attention_output_weight: &self.attention_output_weight,
            post_attention_norm_weight: &self.post_attention_norm_weight,
            gate_weight: &self.gate_weight,
            up_weight: &self.up_weight,
            down_weight: &self.down_weight,
        }
    }
}

#[derive(Clone, Debug)]
struct LayerLengths {
    input: usize,
    query_width: usize,
    key_value_width: usize,
    query_weight: usize,
    key_weight: usize,
    attention_output_weight: usize,
    intermediate_weight: usize,
    down_weight: usize,
}

#[derive(Clone, Debug)]
struct ManualPrefillChain {
    layer_outputs: Vec<Vec<f32>>,
    kv_caches: Vec<DecoderPrefillKvCache>,
    final_norm: Vec<f32>,
}

#[derive(Clone, Debug)]
struct ExpectedCheckpoint {
    layer_index: usize,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct ExpectedDecodeStack {
    checkpoints: Vec<ExpectedCheckpoint>,
    kv_caches: Vec<DecoderPrefillKvCache>,
    final_norm: Vec<f32>,
    executed_layers: usize,
    retained_checkpoint_elements: usize,
    retained_kv_elements: usize,
    layer_inputs: Vec<Vec<f32>>,
    layer_outputs: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct DecodeScenario {
    input: Vec<f32>,
    config: DecoderStackConfig,
    full_tokens: usize,
    prefix_tokens: usize,
    prefix_caches: Vec<DecoderPrefillKvCache>,
    full_caches: Vec<DecoderPrefillKvCache>,
    full_last_rows: Vec<Vec<f32>>,
    full_final_norm_last_row: Vec<f32>,
    final_norm_weight: Vec<f32>,
    parameters: Vec<OwnedParameters>,
    decode_tables: Vec<(Vec<f32>, Vec<f32>)>,
}

trait StackDigestView {
    fn checkpoint_count(&self) -> usize;
    fn checkpoint_at(&self, index: usize) -> (usize, &[f32]);
    fn caches(&self) -> &[DecoderPrefillKvCache];
    fn final_norm(&self) -> &[f32];
    fn executed_layers(&self) -> usize;
    fn retained_checkpoint_elements(&self) -> usize;
    fn retained_kv_elements(&self) -> usize;
}

impl StackDigestView for ExpectedDecodeStack {
    fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    fn checkpoint_at(&self, index: usize) -> (usize, &[f32]) {
        let checkpoint = &self.checkpoints[index];
        (checkpoint.layer_index, &checkpoint.values)
    }

    fn caches(&self) -> &[DecoderPrefillKvCache] {
        &self.kv_caches
    }

    fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }

    fn executed_layers(&self) -> usize {
        self.executed_layers
    }

    fn retained_checkpoint_elements(&self) -> usize {
        self.retained_checkpoint_elements
    }

    fn retained_kv_elements(&self) -> usize {
        self.retained_kv_elements
    }
}

impl StackDigestView for DecoderStackDecodeTrace {
    fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    fn checkpoint_at(&self, index: usize) -> (usize, &[f32]) {
        let checkpoint: &DecoderStackCheckpoint = &self.checkpoints[index];
        (checkpoint.layer_index, &checkpoint.values)
    }

    fn caches(&self) -> &[DecoderPrefillKvCache] {
        &self.kv_caches
    }

    fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }

    fn executed_layers(&self) -> usize {
        self.executed_layers
    }

    fn retained_checkpoint_elements(&self) -> usize {
        self.retained_checkpoint_elements
    }

    fn retained_kv_elements(&self) -> usize {
        self.retained_kv_elements
    }
}

#[derive(Clone, Copy, Debug)]
enum CallbackFault {
    OutputShort,
    OutputLong,
    OutputNonFinite,
    CacheTokensShort,
    CacheTokensLong,
    CacheKeyValueHeadsWrong,
    CacheHeadDimWrong,
    CacheKeysShort,
    CacheKeysLong,
    CacheValuesShort,
    CacheValuesLong,
    CacheKeysNonFinite,
    CacheValuesNonFinite,
}

#[derive(Clone, Copy, Debug)]
enum StackShapeFault {
    InputShort,
    InputLong,
    FinalNormShort,
    FinalNormLong,
}

fn layer_config(tokens: usize) -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        rms_norm_epsilon: EPSILON,
        mrope_sections: SECTIONS,
    }
}

fn stack_config(layers: usize) -> DecoderStackConfig {
    DecoderStackConfig {
        layer: layer_config(1),
        layers,
    }
}

fn pinned_layer_config() -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens: 1,
        hidden_size: PINNED_HIDDEN,
        intermediate_size: PINNED_INTERMEDIATE,
        query_heads: PINNED_QUERY_HEADS,
        key_value_heads: PINNED_KEY_VALUE_HEADS,
        head_dim: PINNED_HEAD_DIM,
        rms_norm_epsilon: PINNED_EPSILON,
        mrope_sections: PINNED_SECTIONS,
    }
}

fn dense(len: usize, multiplier: usize, addend: usize, modulus: usize, divisor: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            (((index * multiplier + addend) % modulus) as f32 - (modulus / 2) as f32) / divisor
        })
        .collect()
}

fn layer_lengths(config: DecoderLayerConfig) -> Option<LayerLengths> {
    let query_width = config.query_heads.checked_mul(config.head_dim)?;
    let key_value_width = config.key_value_heads.checked_mul(config.head_dim)?;
    Some(LayerLengths {
        input: config.tokens.checked_mul(config.hidden_size)?,
        query_width,
        key_value_width,
        query_weight: query_width.checked_mul(config.hidden_size)?,
        key_weight: key_value_width.checked_mul(config.hidden_size)?,
        attention_output_weight: config.hidden_size.checked_mul(query_width)?,
        intermediate_weight: config.intermediate_size.checked_mul(config.hidden_size)?,
        down_weight: config.hidden_size.checked_mul(config.intermediate_size)?,
    })
}

fn token_major_input(tokens: usize, hidden_size: usize) -> Vec<f32> {
    dense(tokens * hidden_size, 17, 3, 29, 11.0)
}

fn one_token_input(hidden_size: usize) -> Vec<f32> {
    dense(hidden_size, 13, 5, 31, 9.0)
}

fn full_input(tokens: usize) -> Vec<f32> {
    token_major_input(tokens, HIDDEN)
}

fn final_norm_weight(hidden_size: usize) -> Vec<f32> {
    (0..hidden_size)
        .map(|index| 0.73 + ((index * 7 + 2) % 13) as f32 / 31.0)
        .collect()
}

fn layer_parameters(layer_index: usize, config: DecoderLayerConfig) -> OwnedParameters {
    let layer = layer_index + 1;
    let lengths = layer_lengths(config).unwrap();
    OwnedParameters {
        input_norm_weight: (0..config.hidden_size)
            .map(|index| 0.71 + ((index * 3 + layer * 2) % 11) as f32 / 37.0)
            .collect(),
        query_weight: dense(
            lengths.query_weight,
            5 + layer * 2,
            3 + layer,
            31 + layer * 2,
            61.0 + layer as f32,
        ),
        key_weight: dense(
            lengths.key_weight,
            7 + layer * 2,
            5 + layer,
            37 + layer * 2,
            67.0 + layer as f32,
        ),
        value_weight: dense(
            lengths.key_weight,
            11 + layer * 2,
            7 + layer,
            41 + layer * 2,
            71.0 + layer as f32,
        ),
        attention_output_weight: dense(
            lengths.attention_output_weight,
            13 + layer * 2,
            2 + layer,
            43 + layer * 2,
            79.0 + layer as f32,
        ),
        post_attention_norm_weight: (0..config.hidden_size)
            .map(|index| 0.67 + ((index * 5 + layer * 3) % 13) as f32 / 41.0)
            .collect(),
        gate_weight: dense(
            lengths.intermediate_weight,
            17 + layer * 2,
            1 + layer,
            47 + layer * 2,
            83.0 + layer as f32,
        ),
        up_weight: dense(
            lengths.intermediate_weight,
            19 + layer * 2,
            4 + layer,
            53 + layer * 2,
            89.0 + layer as f32,
        ),
        down_weight: dense(
            lengths.down_weight,
            23 + layer * 2,
            6 + layer,
            59 + layer * 2,
            97.0 + layer as f32,
        ),
    }
}

fn raw_tables(layer_index: usize, tokens: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = Vec::with_capacity(3 * tokens * head_dim);
    let mut sine = Vec::with_capacity(3 * tokens * head_dim);
    for axis in 0..3 {
        for token in 0..tokens {
            for dimension in 0..head_dim {
                cosine.push(
                    0.51 + layer_index as f32 * 0.017
                        + axis as f32 * 0.11
                        + token as f32 * 0.019
                        + dimension as f32 * 0.006,
                );
                sine.push(
                    -0.27 - layer_index as f32 * 0.013 - axis as f32 * 0.09 + token as f32 * 0.014
                        - dimension as f32 * 0.004,
                );
            }
        }
    }
    (cosine, sine)
}

fn slice_row(values: &[f32], row: usize, rows: usize, width: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * width);
    let start = row * width;
    values[start..start + width].to_vec()
}

fn slice_last_row(values: &[f32], rows: usize, width: usize) -> Vec<f32> {
    slice_row(values, rows - 1, rows, width)
}

fn slice_prefix_rows(values: &[f32], rows: usize, prefix_rows: usize, width: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * width);
    values[..prefix_rows * width].to_vec()
}

fn slice_raw_axis_major_row(values: &[f32], rows: usize, row: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(values.len(), 3 * rows * head_dim);
    let mut output = Vec::with_capacity(3 * head_dim);
    for axis in 0..3 {
        let start = axis * rows * head_dim + row * head_dim;
        output.extend_from_slice(&values[start..start + head_dim]);
    }
    output
}

fn slice_last_raw_axis_major_row(values: &[f32], rows: usize, head_dim: usize) -> Vec<f32> {
    slice_raw_axis_major_row(values, rows, rows - 1, head_dim)
}

fn slice_prefix_raw_axis_major(
    values: &[f32],
    rows: usize,
    prefix_rows: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(values.len(), 3 * rows * head_dim);
    let mut output = Vec::with_capacity(3 * prefix_rows * head_dim);
    for axis in 0..3 {
        let start = axis * rows * head_dim;
        output.extend_from_slice(&values[start..start + prefix_rows * head_dim]);
    }
    output
}

fn stage_values(len: usize, layer_index: usize, stage: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let signed = ((index * 7 + layer_index * 11 + stage * 5) % 29) as f32 - 14.0;
            signed / 113.0
        })
        .collect()
}

fn next_output(layer_index: usize, input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let signed_channel = (index % 7) as f32 - 3.0;
            value + (layer_index + 1) as f32 * 0.000_37 * signed_channel
        })
        .collect()
}

fn make_valid_cache(
    tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    layer_index: usize,
) -> DecoderPrefillKvCache {
    let width = key_value_heads
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(tokens))
        .unwrap();
    DecoderPrefillKvCache {
        keys: stage_values(width, layer_index, 20),
        values: stage_values(width, layer_index, 21),
        tokens,
        key_value_heads,
        head_dim,
    }
}

fn valid_decode_trace(
    config: DecoderLayerConfig,
    layer_index: usize,
    prefix_tokens: usize,
    output: Vec<f32>,
) -> DecoderLayerDecodeTrace {
    let lengths = layer_lengths(config).unwrap();
    assert_eq!(output.len(), lengths.input);
    let intermediate = config.tokens.checked_mul(config.intermediate_size).unwrap();
    let full_cache_tokens = prefix_tokens + 1;
    let full_cache_len = full_cache_tokens
        .checked_mul(lengths.key_value_width)
        .unwrap();
    DecoderLayerDecodeTrace {
        norm1: stage_values(lengths.input, layer_index, 0),
        query: stage_values(lengths.query_width, layer_index, 1),
        key: stage_values(lengths.key_value_width, layer_index, 2),
        value: stage_values(lengths.key_value_width, layer_index, 3),
        mrope_query: stage_values(lengths.query_width, layer_index, 4),
        mrope_key: stage_values(lengths.key_value_width, layer_index, 5),
        kv_cache: DecoderPrefillKvCache {
            keys: stage_values(full_cache_len, layer_index, 6),
            values: stage_values(full_cache_len, layer_index, 7),
            tokens: full_cache_tokens,
            key_value_heads: config.key_value_heads,
            head_dim: config.head_dim,
        },
        attention_context: stage_values(lengths.query_width, layer_index, 8),
        attention_output: stage_values(lengths.input, layer_index, 9),
        attention_residual: stage_values(lengths.input, layer_index, 10),
        norm2: stage_values(lengths.input, layer_index, 11),
        mlp_gate: stage_values(intermediate, layer_index, 12),
        mlp_up: stage_values(intermediate, layer_index, 13),
        mlp_activation: stage_values(intermediate, layer_index, 14),
        mlp_down: stage_values(lengths.input, layer_index, 15),
        output,
    }
}

fn synthetic_expected_decode_stack(
    input: &[f32],
    config: DecoderStackConfig,
    prefix_tokens: usize,
    checkpoint_layers: &[usize],
) -> ExpectedDecodeStack {
    let mut current = input.to_vec();
    let mut layer_inputs = Vec::with_capacity(config.layers);
    let mut layer_outputs = Vec::with_capacity(config.layers);
    let mut checkpoints = Vec::new();
    let mut kv_caches = Vec::with_capacity(config.layers);
    let mut checkpoint_cursor = 0_usize;
    for layer_index in 0..config.layers {
        layer_inputs.push(current.clone());
        let trace = valid_decode_trace(
            config.layer,
            layer_index,
            prefix_tokens,
            next_output(layer_index, &current),
        );
        current = trace.output.clone();
        layer_outputs.push(trace.output.clone());
        if checkpoint_layers.get(checkpoint_cursor) == Some(&layer_index) {
            checkpoints.push(ExpectedCheckpoint {
                layer_index,
                values: trace.output.clone(),
            });
            checkpoint_cursor += 1;
        }
        kv_caches.push(trace.kv_cache);
    }
    let final_norm = rms_norm_f32(
        &current,
        config.layer.tokens,
        config.layer.hidden_size,
        &final_norm_weight(config.layer.hidden_size),
        config.layer.rms_norm_epsilon,
    )
    .unwrap();
    let retained_checkpoint_elements = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.values.len())
        .sum();
    let retained_kv_elements = kv_caches
        .iter()
        .map(|cache| cache.keys.len() + cache.values.len())
        .sum();
    ExpectedDecodeStack {
        checkpoints,
        kv_caches,
        final_norm,
        executed_layers: config.layers,
        retained_checkpoint_elements,
        retained_kv_elements,
        layer_inputs,
        layer_outputs,
    }
}

fn trace_with_liveness_markers(
    config: DecoderLayerConfig,
    layer_index: usize,
    prefix_tokens: usize,
    output: Vec<f32>,
) -> DecoderLayerDecodeTrace {
    let mut trace = valid_decode_trace(config, layer_index, prefix_tokens, output);
    let marker = layer_index as f32 + 0.25;
    trace.norm1 = vec![marker; LIVENESS_MARKER_FLOATS];
    trace.query = vec![marker + 0.1; LIVENESS_MARKER_FLOATS];
    trace.attention_context = vec![marker + 0.2; LIVENESS_MARKER_FLOATS];
    trace.mlp_activation = vec![marker + 0.3; LIVENESS_MARKER_FLOATS];
    trace
}

fn mutate_trace_fault(
    trace: &mut DecoderLayerDecodeTrace,
    fault: CallbackFault,
    prefix_tokens: usize,
    config: DecoderLayerConfig,
) {
    match fault {
        CallbackFault::OutputShort => {
            trace.output.pop();
        }
        CallbackFault::OutputLong => {
            trace.output.push(0.0);
        }
        CallbackFault::OutputNonFinite => {
            trace.output[config.hidden_size / 2] = f32::NAN;
        }
        CallbackFault::CacheTokensShort => {
            trace.kv_cache.tokens = prefix_tokens;
        }
        CallbackFault::CacheTokensLong => {
            trace.kv_cache.tokens = prefix_tokens + 2;
        }
        CallbackFault::CacheKeyValueHeadsWrong => {
            trace.kv_cache.key_value_heads += 1;
        }
        CallbackFault::CacheHeadDimWrong => {
            trace.kv_cache.head_dim += 2;
        }
        CallbackFault::CacheKeysShort => {
            trace.kv_cache.keys.pop();
        }
        CallbackFault::CacheKeysLong => {
            trace.kv_cache.keys.push(0.0);
        }
        CallbackFault::CacheValuesShort => {
            trace.kv_cache.values.pop();
        }
        CallbackFault::CacheValuesLong => {
            trace.kv_cache.values.push(0.0);
        }
        CallbackFault::CacheKeysNonFinite => {
            trace.kv_cache.keys[0] = f32::INFINITY;
        }
        CallbackFault::CacheValuesNonFinite => {
            let last = trace.kv_cache.values.len() - 1;
            trace.kv_cache.values[last] = f32::NEG_INFINITY;
        }
    }
}

fn manual_prefill_chain(
    input: &[f32],
    tokens: usize,
    layers: usize,
    final_norm_weight: &[f32],
    tables: &[(Vec<f32>, Vec<f32>)],
    parameters: &[OwnedParameters],
) -> ManualPrefillChain {
    let mut current = input.to_vec();
    let mut layer_outputs = Vec::with_capacity(layers);
    let mut kv_caches = Vec::with_capacity(layers);
    for layer_index in 0..layers {
        let config = layer_config(tokens);
        let trace = decoder_layer_prefill_f32(
            &current,
            config,
            &tables[layer_index].0,
            &tables[layer_index].1,
            parameters[layer_index].borrowed(),
        )
        .unwrap();
        current = trace.output.clone();
        layer_outputs.push(trace.output);
        kv_caches.push(trace.kv_cache);
    }
    let final_norm = rms_norm_f32(
        layer_outputs.last().unwrap(),
        tokens,
        HIDDEN,
        final_norm_weight,
        EPSILON,
    )
    .unwrap();
    ManualPrefillChain {
        layer_outputs,
        kv_caches,
        final_norm,
    }
}

fn manual_decode_stack(
    input: &[f32],
    config: DecoderStackConfig,
    prefix_caches: &[DecoderPrefillKvCache],
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    tables: &[(Vec<f32>, Vec<f32>)],
    parameters: &[OwnedParameters],
) -> ExpectedDecodeStack {
    let mut current = input.to_vec();
    let mut layer_inputs = Vec::with_capacity(config.layers);
    let mut layer_outputs = Vec::with_capacity(config.layers);
    let mut checkpoints = Vec::new();
    let mut kv_caches = Vec::with_capacity(config.layers);
    let mut checkpoint_cursor = 0_usize;
    for layer_index in 0..config.layers {
        layer_inputs.push(current.clone());
        let trace = decoder_layer_decode_f32(
            &current,
            config.layer,
            &tables[layer_index].0,
            &tables[layer_index].1,
            &prefix_caches[layer_index],
            parameters[layer_index].borrowed(),
        )
        .unwrap();
        current = trace.output.clone();
        layer_outputs.push(trace.output.clone());
        if checkpoint_layers.get(checkpoint_cursor) == Some(&layer_index) {
            checkpoints.push(ExpectedCheckpoint {
                layer_index,
                values: trace.output.clone(),
            });
            checkpoint_cursor += 1;
        }
        kv_caches.push(trace.kv_cache);
    }
    let final_norm = rms_norm_f32(
        &current,
        config.layer.tokens,
        config.layer.hidden_size,
        final_norm_weight,
        config.layer.rms_norm_epsilon,
    )
    .unwrap();
    let retained_checkpoint_elements = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.values.len())
        .sum();
    let retained_kv_elements = kv_caches
        .iter()
        .map(|cache| cache.keys.len() + cache.values.len())
        .sum();
    ExpectedDecodeStack {
        checkpoints,
        kv_caches,
        final_norm,
        executed_layers: config.layers,
        retained_checkpoint_elements,
        retained_kv_elements,
        layer_inputs,
        layer_outputs,
    }
}

fn decode_plan_digest(
    scenario: &DecodeScenario,
    order: &[usize],
    reset_to_original: bool,
    wrong_cache_rotation: bool,
) -> String {
    let mut current = scenario.input.clone();
    let mut hasher = blake3::Hasher::new();
    update_usize_digest(&mut hasher, order.len());
    for &layer_index in order {
        let current_input = if reset_to_original {
            scenario.input.as_slice()
        } else {
            &current
        };
        let cache_index = if wrong_cache_rotation {
            (layer_index + 1) % scenario.prefix_caches.len()
        } else {
            layer_index
        };
        let trace = decoder_layer_decode_f32(
            current_input,
            scenario.config.layer,
            &scenario.decode_tables[layer_index].0,
            &scenario.decode_tables[layer_index].1,
            &scenario.prefix_caches[cache_index],
            scenario.parameters[layer_index].borrowed(),
        )
        .unwrap();
        update_usize_digest(&mut hasher, layer_index);
        update_f32_digest(&mut hasher, &trace.output);
        update_f32_digest(&mut hasher, &trace.kv_cache.keys);
        update_f32_digest(&mut hasher, &trace.kv_cache.values);
        current = trace.output;
    }
    let final_norm = rms_norm_f32(
        &current,
        scenario.config.layer.tokens,
        scenario.config.layer.hidden_size,
        &scenario.final_norm_weight,
        scenario.config.layer.rms_norm_epsilon,
    )
    .unwrap();
    update_f32_digest(&mut hasher, &final_norm);
    hasher.finalize().to_hex().to_string()
}

fn build_decode_scenario(
    prefix_tokens: usize,
    layers: usize,
    checkpoint_layers: &[usize],
) -> DecodeScenario {
    let full_tokens = prefix_tokens + 1;
    let config = DecoderStackConfig {
        layer: layer_config(1),
        layers,
    };
    let full_input = full_input(full_tokens);
    let input = slice_last_row(&full_input, full_tokens, HIDDEN);
    let prefix_input = slice_prefix_rows(&full_input, full_tokens, prefix_tokens, HIDDEN);
    let final_norm_weight = final_norm_weight(HIDDEN);
    let parameters = (0..layers)
        .map(|layer_index| layer_parameters(layer_index, config.layer))
        .collect::<Vec<_>>();
    let full_tables = (0..layers)
        .map(|layer_index| raw_tables(layer_index, full_tokens, HEAD_DIM))
        .collect::<Vec<_>>();
    let prefix_tables = full_tables
        .iter()
        .map(|(cosine, sine)| {
            (
                slice_prefix_raw_axis_major(cosine, full_tokens, prefix_tokens, HEAD_DIM),
                slice_prefix_raw_axis_major(sine, full_tokens, prefix_tokens, HEAD_DIM),
            )
        })
        .collect::<Vec<_>>();
    let decode_tables = full_tables
        .iter()
        .map(|(cosine, sine)| {
            (
                slice_last_raw_axis_major_row(cosine, full_tokens, HEAD_DIM),
                slice_last_raw_axis_major_row(sine, full_tokens, HEAD_DIM),
            )
        })
        .collect::<Vec<_>>();
    let full_chain = manual_prefill_chain(
        &full_input,
        full_tokens,
        layers,
        &final_norm_weight,
        &full_tables,
        &parameters,
    );
    let prefix_chain = manual_prefill_chain(
        &prefix_input,
        prefix_tokens,
        layers,
        &final_norm_weight,
        &prefix_tables,
        &parameters,
    );
    let expected = manual_decode_stack(
        &input,
        config,
        &prefix_chain.kv_caches,
        checkpoint_layers,
        &final_norm_weight,
        &decode_tables,
        &parameters,
    );
    for layer_index in 0..layers {
        assert_f32_bits(
            &format!("full-vs-cache output layer {layer_index}"),
            &expected.layer_outputs[layer_index],
            &slice_last_row(&full_chain.layer_outputs[layer_index], full_tokens, HIDDEN),
        );
        assert_cache_exact(
            &format!("full-vs-cache cache layer {layer_index}"),
            &expected.kv_caches[layer_index],
            &full_chain.kv_caches[layer_index],
        );
    }
    assert_f32_bits(
        "full-vs-cache final norm",
        &expected.final_norm,
        &slice_last_row(&full_chain.final_norm, full_tokens, HIDDEN),
    );
    DecodeScenario {
        input,
        config,
        full_tokens,
        prefix_tokens,
        prefix_caches: prefix_chain.kv_caches,
        full_caches: full_chain.kv_caches,
        full_last_rows: full_chain
            .layer_outputs
            .iter()
            .map(|values| slice_last_row(values, full_tokens, HIDDEN))
            .collect(),
        full_final_norm_last_row: slice_last_row(&full_chain.final_norm, full_tokens, HIDDEN),
        final_norm_weight,
        parameters,
        decode_tables,
    }
}

fn update_usize_digest(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}

fn update_f32_digest(hasher: &mut blake3::Hasher, values: &[f32]) {
    update_usize_digest(hasher, values.len());
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn combined_digest(trace: &impl StackDigestView) -> String {
    let mut hasher = blake3::Hasher::new();
    update_usize_digest(&mut hasher, trace.checkpoint_count());
    for index in 0..trace.checkpoint_count() {
        let (layer_index, values) = trace.checkpoint_at(index);
        update_usize_digest(&mut hasher, layer_index);
        update_f32_digest(&mut hasher, values);
    }
    update_usize_digest(&mut hasher, trace.caches().len());
    for cache in trace.caches() {
        update_usize_digest(&mut hasher, cache.tokens);
        update_usize_digest(&mut hasher, cache.key_value_heads);
        update_usize_digest(&mut hasher, cache.head_dim);
        update_f32_digest(&mut hasher, &cache.keys);
        update_f32_digest(&mut hasher, &cache.values);
    }
    update_f32_digest(&mut hasher, trace.final_norm());
    update_usize_digest(&mut hasher, trace.executed_layers());
    update_usize_digest(&mut hasher, trace.retained_checkpoint_elements());
    update_usize_digest(&mut hasher, trace.retained_kv_elements());
    hasher.finalize().to_hex().to_string()
}

fn assert_f32_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}]: actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_cache_exact(
    label: &str,
    actual: &DecoderPrefillKvCache,
    expected: &DecoderPrefillKvCache,
) {
    assert_eq!(actual.tokens, expected.tokens, "{label} tokens");
    assert_eq!(
        actual.key_value_heads, expected.key_value_heads,
        "{label} key_value_heads"
    );
    assert_eq!(actual.head_dim, expected.head_dim, "{label} head_dim");
    assert_f32_bits(&format!("{label} keys"), &actual.keys, &expected.keys);
    assert_f32_bits(&format!("{label} values"), &actual.values, &expected.values);
}

fn assert_stack_exact(actual: &DecoderStackDecodeTrace, expected: &ExpectedDecodeStack) {
    assert_eq!(actual.checkpoints.len(), expected.checkpoints.len());
    for (index, checkpoint) in expected.checkpoints.iter().enumerate() {
        let actual_checkpoint: &DecoderStackCheckpoint = &actual.checkpoints[index];
        assert_eq!(actual_checkpoint.layer_index, checkpoint.layer_index);
        assert_f32_bits(
            &format!("checkpoint {}", checkpoint.layer_index),
            &actual_checkpoint.values,
            &checkpoint.values,
        );
    }
    assert_eq!(actual.kv_caches.len(), expected.kv_caches.len());
    for (layer_index, (actual_cache, expected_cache)) in
        actual.kv_caches.iter().zip(&expected.kv_caches).enumerate()
    {
        assert_cache_exact(
            &format!("cache {layer_index}"),
            actual_cache,
            expected_cache,
        );
    }
    assert_f32_bits("final norm", &actual.final_norm, &expected.final_norm);
    assert_eq!(actual.executed_layers, expected.executed_layers);
    assert_eq!(
        actual.retained_checkpoint_elements,
        expected.retained_checkpoint_elements
    );
    assert_eq!(actual.retained_kv_elements, expected.retained_kv_elements);
}

fn assert_error<T>(case: &str, result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{case}"),
        Ok(_) => panic!("{case}: expected {expected:?}"),
    }
}

fn assert_rejected_before_layer_zero(
    case: &str,
    input: &[f32],
    config: DecoderStackConfig,
    prefix_caches: &[DecoderPrefillKvCache],
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    expected: CpuRefErrorCode,
) {
    let mut calls = 0_usize;
    let result = decoder_stack_decode_f32(
        input,
        config,
        prefix_caches,
        checkpoint_layers,
        final_norm_weight,
        |_, _, _, _| -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            calls += 1;
            panic!("{case}: invalid decode stack invoked callback zero")
        },
    );
    assert_eq!(calls, 0, "{case}");
    assert_error(case, result, expected);
}

fn assert_pinned_rejected_before_layer_zero(
    case: &str,
    input: &[f32],
    prefix_caches: &[DecoderPrefillKvCache],
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    expected: CpuRefErrorCode,
) {
    let mut calls = 0_usize;
    let result = pinned_decoder_stack_decode_f32(
        input,
        prefix_caches,
        checkpoint_layers,
        final_norm_weight,
        |_, _, _, _| -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            calls += 1;
            panic!("{case}: invalid pinned decode stack invoked callback zero")
        },
    );
    assert_eq!(calls, 0, "{case}");
    assert_error(case, result, expected);
}

fn shape_consistent_input(config: DecoderLayerConfig, fault: StackShapeFault) -> Vec<f32> {
    let input_len = config.tokens.checked_mul(config.hidden_size).unwrap();
    let mut values = dense(input_len, 19, 4, 37, 17.0);
    match fault {
        StackShapeFault::InputShort => {
            values.pop();
        }
        StackShapeFault::InputLong => {
            values.push(0.0);
        }
        StackShapeFault::FinalNormShort | StackShapeFault::FinalNormLong => {}
    }
    values
}

fn shape_consistent_final_norm(config: DecoderLayerConfig, fault: StackShapeFault) -> Vec<f32> {
    let mut values = final_norm_weight(config.hidden_size);
    match fault {
        StackShapeFault::FinalNormShort => {
            values.pop();
        }
        StackShapeFault::FinalNormLong => {
            values.push(0.0);
        }
        StackShapeFault::InputShort | StackShapeFault::InputLong => {}
    }
    values
}

#[test]
fn real_tiny_decode_stack_matches_independent_oracle_literal_anchor_and_preserves_operands() {
    let scenario = build_decode_scenario(3, LAYERS, &CHECKPOINTS);
    let expected = manual_decode_stack(
        &scenario.input,
        scenario.config,
        &scenario.prefix_caches,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        &scenario.decode_tables,
        &scenario.parameters,
    );
    assert_eq!(combined_digest(&expected), COMPACT_DECODE_STACK_BLAKE3);
    for layer_index in 0..LAYERS {
        assert_f32_bits(
            &format!("tiny full last row layer {layer_index}"),
            &expected.layer_outputs[layer_index],
            &scenario.full_last_rows[layer_index],
        );
        assert_cache_exact(
            &format!("tiny full cache layer {layer_index}"),
            &expected.kv_caches[layer_index],
            &scenario.full_caches[layer_index],
        );
    }
    assert_f32_bits(
        "tiny full final norm last row",
        &expected.final_norm,
        &scenario.full_final_norm_last_row,
    );

    let preserved = (
        scenario.input.clone(),
        scenario.prefix_caches.clone(),
        scenario.parameters.clone(),
        scenario.decode_tables.clone(),
        scenario.final_norm_weight.clone(),
    );
    let mut invocation_order = Vec::new();
    let mut observed_inputs = Vec::new();
    let mut observed_prefix_caches = Vec::new();
    let mut actual = decoder_stack_decode_f32(
        &scenario.input,
        scenario.config,
        &scenario.prefix_caches,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            invocation_order.push(layer_index);
            observed_inputs.push(current_one_token.to_vec());
            observed_prefix_caches.push(prefix_cache.clone());
            assert_eq!(supplied_config, scenario.config.layer);
            decoder_layer_decode_f32(
                current_one_token,
                supplied_config,
                &scenario.decode_tables[layer_index].0,
                &scenario.decode_tables[layer_index].1,
                prefix_cache,
                scenario.parameters[layer_index].borrowed(),
            )
        },
    )
    .unwrap();

    assert_eq!(invocation_order, [0, 1, 2, 3]);
    assert_eq!(observed_inputs, expected.layer_inputs);
    assert_eq!(observed_prefix_caches.len(), scenario.prefix_caches.len());
    for (layer_index, observed_cache) in observed_prefix_caches.iter().enumerate() {
        assert_cache_exact(
            &format!("observed prefix cache {layer_index}"),
            observed_cache,
            &scenario.prefix_caches[layer_index],
        );
    }
    assert_stack_exact(&actual, &expected);
    assert_eq!(combined_digest(&actual), COMPACT_DECODE_STACK_BLAKE3);
    for checkpoint in &expected.checkpoints {
        assert_f32_bits(
            &format!("checkpoint accessor {}", checkpoint.layer_index),
            actual.checkpoint(checkpoint.layer_index).unwrap(),
            &checkpoint.values,
        );
    }
    assert_eq!(actual.checkpoint(1), None);
    assert_eq!(actual.checkpoint(LAYERS), None);
    for layer_index in 0..LAYERS {
        assert_cache_exact(
            &format!("cache accessor {layer_index}"),
            actual.kv_cache(layer_index).unwrap(),
            &expected.kv_caches[layer_index],
        );
    }
    assert_eq!(actual.kv_cache(LAYERS), None);
    assert_eq!(scenario.input, preserved.0);
    assert_eq!(scenario.prefix_caches, preserved.1);
    assert_eq!(scenario.parameters, preserved.2);
    assert_eq!(scenario.decode_tables, preserved.3);
    assert_eq!(scenario.final_norm_weight, preserved.4);

    let preserved_prefix = scenario.prefix_caches.clone();
    let preserved_output_cache = actual.kv_caches[0].clone();
    actual.kv_caches[0].keys[0] = 123.5;
    actual.kv_caches[0].values[1] = -456.25;
    assert_eq!(scenario.prefix_caches, preserved_prefix);
    assert_eq!(expected.kv_caches[0], preserved_output_cache);
}

#[test]
fn decode_stack_matches_full_recompute_across_prefix_boundaries_while_logits_top_k_generation_and_random_splits_wait_for_later_m6()
 {
    for prefix_tokens in PREFIX_BOUNDARIES {
        let checkpoint_layers = [0, 1, 2, 3];
        let scenario = build_decode_scenario(prefix_tokens, LAYERS, &checkpoint_layers);
        assert_eq!(scenario.prefix_tokens, prefix_tokens);
        assert_eq!(
            scenario.full_tokens,
            prefix_tokens
                .checked_add(1)
                .expect("prefix boundary overflow in compact test fixture"),
        );
        assert_eq!(scenario.full_tokens, scenario.prefix_tokens + 1);
        let expected = manual_decode_stack(
            &scenario.input,
            scenario.config,
            &scenario.prefix_caches,
            &checkpoint_layers,
            &scenario.final_norm_weight,
            &scenario.decode_tables,
            &scenario.parameters,
        );
        let actual = decoder_stack_decode_f32(
            &scenario.input,
            scenario.config,
            &scenario.prefix_caches,
            &checkpoint_layers,
            &scenario.final_norm_weight,
            |layer_index: usize,
             supplied_config: DecoderLayerConfig,
             current_one_token: &[f32],
             prefix_cache: &DecoderPrefillKvCache|
             -> Result<DecoderLayerDecodeTrace, CpuRefError> {
                decoder_layer_decode_f32(
                    current_one_token,
                    supplied_config,
                    &scenario.decode_tables[layer_index].0,
                    &scenario.decode_tables[layer_index].1,
                    prefix_cache,
                    scenario.parameters[layer_index].borrowed(),
                )
            },
        )
        .unwrap();

        assert_stack_exact(&actual, &expected);
        for layer_index in 0..LAYERS {
            assert_f32_bits(
                &format!("prefix_tokens={prefix_tokens} full last row output {layer_index}"),
                actual.checkpoint(layer_index).unwrap(),
                &scenario.full_last_rows[layer_index],
            );
            assert_cache_exact(
                &format!("prefix_tokens={prefix_tokens} cache {layer_index}"),
                actual.kv_cache(layer_index).unwrap(),
                &scenario.full_caches[layer_index],
            );
        }
        assert_f32_bits(
            &format!("prefix_tokens={prefix_tokens} final norm"),
            &actual.final_norm,
            &scenario.full_final_norm_last_row,
        );
    }
}

#[test]
fn decode_stack_rejects_invalid_stack_contracts_before_layer_zero() {
    let scenario = build_decode_scenario(3, LAYERS, &CHECKPOINTS);
    assert_rejected_before_layer_zero(
        "zero layers",
        &scenario.input,
        stack_config(0),
        &scenario.prefix_caches,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );
    for tokens in [0, 2] {
        let config = DecoderStackConfig {
            layer: DecoderLayerConfig {
                tokens,
                ..scenario.config.layer
            },
            layers: LAYERS,
        };
        let shape_consistent_input = dense(
            config.layer.tokens * config.layer.hidden_size,
            19,
            4,
            37,
            17.0,
        );
        let shape_consistent_final_norm = final_norm_weight(config.layer.hidden_size);
        assert_rejected_before_layer_zero(
            &format!("config.layer.tokens={tokens} shape-consistent"),
            &shape_consistent_input,
            config,
            &scenario.prefix_caches,
            &CHECKPOINTS,
            &shape_consistent_final_norm,
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for fault in [
        StackShapeFault::InputShort,
        StackShapeFault::InputLong,
        StackShapeFault::FinalNormShort,
        StackShapeFault::FinalNormLong,
    ] {
        assert_rejected_before_layer_zero(
            &format!("shape fault {fault:?}"),
            &shape_consistent_input(scenario.config.layer, fault),
            scenario.config,
            &scenario.prefix_caches,
            &CHECKPOINTS,
            &shape_consistent_final_norm(scenario.config.layer, fault),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for checkpoint_layers in [vec![2, 1], vec![1, 1], vec![0, LAYERS]] {
        assert_rejected_before_layer_zero(
            &format!("invalid checkpoints {checkpoint_layers:?}"),
            &scenario.input,
            scenario.config,
            &scenario.prefix_caches,
            &checkpoint_layers,
            &scenario.final_norm_weight,
            CpuRefErrorCode::InvalidCheckpointSelection,
        );
    }
    let mut too_few_prefix = scenario.prefix_caches.clone();
    too_few_prefix.pop();
    assert_rejected_before_layer_zero(
        "prefix cache count short",
        &scenario.input,
        scenario.config,
        &too_few_prefix,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );
    let mut too_many_prefix = scenario.prefix_caches.clone();
    too_many_prefix.push(scenario.prefix_caches[0].clone());
    assert_rejected_before_layer_zero(
        "prefix cache count long",
        &scenario.input,
        scenario.config,
        &too_many_prefix,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );

    let malformed_prefix_cases = [
        ("prefix cache zero tokens", {
            let mut caches = scenario.prefix_caches.clone();
            caches[1] = DecoderPrefillKvCache {
                keys: Vec::new(),
                values: Vec::new(),
                tokens: 0,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            };
            caches
        }),
        ("prefix cache keys short", {
            let mut caches = scenario.prefix_caches.clone();
            caches[2].keys.pop();
            caches
        }),
        ("prefix cache values long", {
            let mut caches = scenario.prefix_caches.clone();
            caches[0].values.push(0.0);
            caches
        }),
        ("prefix cache metadata mismatch", {
            let mut caches = scenario.prefix_caches.clone();
            caches[3].head_dim += 2;
            caches
        }),
        ("prefix cache key_value_heads mismatch", {
            let mut caches = scenario.prefix_caches.clone();
            caches[0].key_value_heads += 1;
            caches
        }),
        ("prefix cache nonfinite", {
            let mut caches = scenario.prefix_caches.clone();
            caches[1].keys[0] = f32::NAN;
            caches
        }),
        ("prefix cache values nonfinite", {
            let mut caches = scenario.prefix_caches.clone();
            let last = caches[2].values.len() - 1;
            caches[2].values[last] = f32::INFINITY;
            caches
        }),
        ("prefix cache inconsistent token counts", {
            let mut caches = scenario.prefix_caches.clone();
            caches[2].tokens -= 1;
            let reduced_tokens = caches[2].tokens;
            let width = KEY_VALUE_HEADS * HEAD_DIM;
            caches[2].keys.truncate(reduced_tokens * width);
            caches[2].values.truncate(reduced_tokens * width);
            caches
        }),
    ];
    for (case, caches) in malformed_prefix_cases {
        assert_rejected_before_layer_zero(
            case,
            &scenario.input,
            scenario.config,
            &caches,
            &CHECKPOINTS,
            &scenario.final_norm_weight,
            if case == "prefix cache nonfinite" || case == "prefix cache values nonfinite" {
                CpuRefErrorCode::NonFiniteInput
            } else {
                CpuRefErrorCode::DimensionMismatch
            },
        );
    }

    for bad_index in [
        0,
        scenario.final_norm_weight.len() / 2,
        scenario.final_norm_weight.len() - 1,
    ] {
        let mut bad = scenario.final_norm_weight.clone();
        bad[bad_index] = f32::INFINITY;
        assert_rejected_before_layer_zero(
            &format!("final norm nonfinite index {bad_index}"),
            &scenario.input,
            scenario.config,
            &scenario.prefix_caches,
            &CHECKPOINTS,
            &bad,
            CpuRefErrorCode::NonFiniteInput,
        );
    }
    for bad_index in [0, scenario.input.len() / 2, scenario.input.len() - 1] {
        let mut bad = scenario.input.clone();
        bad[bad_index] = f32::NEG_INFINITY;
        assert_rejected_before_layer_zero(
            &format!("input nonfinite index {bad_index}"),
            &bad,
            scenario.config,
            &scenario.prefix_caches,
            &CHECKPOINTS,
            &scenario.final_norm_weight,
            CpuRefErrorCode::NonFiniteInput,
        );
    }

    assert_rejected_before_layer_zero(
        "query width checked overflow with shape-consistent prefix cache metadata",
        &scenario.input,
        DecoderStackConfig {
            layer: DecoderLayerConfig {
                query_heads: usize::MAX - 1,
                key_value_heads: 2,
                head_dim: 6,
                mrope_sections: [1, 1, 1],
                ..scenario.config.layer
            },
            layers: LAYERS,
        },
        &scenario.prefix_caches,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );
    let mut overflowing_prefix_tokens = scenario.prefix_caches.clone();
    overflowing_prefix_tokens[0] = DecoderPrefillKvCache {
        keys: Vec::new(),
        values: Vec::new(),
        tokens: usize::MAX,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
    };
    assert_rejected_before_layer_zero(
        "prefix cache token arithmetic overflow before callback zero",
        &scenario.input,
        scenario.config,
        &overflowing_prefix_tokens,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );
}

#[test]
fn decode_stack_validates_callback_results_before_next_layer_and_propagates_errors() {
    let scenario = build_decode_scenario(3, LAYERS, &CHECKPOINTS);
    for fault in [
        CallbackFault::OutputShort,
        CallbackFault::OutputLong,
        CallbackFault::OutputNonFinite,
        CallbackFault::CacheTokensShort,
        CallbackFault::CacheTokensLong,
        CallbackFault::CacheKeyValueHeadsWrong,
        CallbackFault::CacheHeadDimWrong,
        CallbackFault::CacheKeysShort,
        CallbackFault::CacheKeysLong,
        CallbackFault::CacheValuesShort,
        CallbackFault::CacheValuesLong,
        CallbackFault::CacheKeysNonFinite,
        CallbackFault::CacheValuesNonFinite,
    ] {
        let bad_layer = 2;
        let mut calls = Vec::new();
        let result = decoder_stack_decode_f32(
            &scenario.input,
            scenario.config,
            &scenario.prefix_caches,
            &CHECKPOINTS,
            &scenario.final_norm_weight,
            |layer_index: usize,
             supplied_config: DecoderLayerConfig,
             current_one_token: &[f32],
             prefix_cache: &DecoderPrefillKvCache|
             -> Result<DecoderLayerDecodeTrace, CpuRefError> {
                calls.push(layer_index);
                let mut trace = valid_decode_trace(
                    supplied_config,
                    layer_index,
                    prefix_cache.tokens,
                    next_output(layer_index, current_one_token),
                );
                if layer_index == bad_layer {
                    mutate_trace_fault(&mut trace, fault, prefix_cache.tokens, supplied_config);
                }
                Ok(trace)
            },
        );
        assert_eq!(calls, vec![0, 1, 2], "{fault:?} producing layer");
        assert_error(
            &format!("callback fault {fault:?}"),
            result,
            match fault {
                CallbackFault::OutputNonFinite
                | CallbackFault::CacheKeysNonFinite
                | CallbackFault::CacheValuesNonFinite => CpuRefErrorCode::NonFiniteInput,
                _ => CpuRefErrorCode::DimensionMismatch,
            },
        );
    }

    let mut calls = Vec::new();
    let result = decoder_stack_decode_f32(
        &scenario.input,
        scenario.config,
        &scenario.prefix_caches,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            calls.push(layer_index);
            if layer_index == 1 {
                return Err(rms_norm_f32(&[f32::NAN], 1, 1, &[1.0], 1.0)
                    .expect_err("forced callback error"));
            }
            Ok(valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            ))
        },
    );
    assert_eq!(calls, vec![0, 1]);
    assert_error(
        "propagated callback error",
        result,
        CpuRefErrorCode::NonFiniteInput,
    );
}

#[test]
fn decode_stack_retains_only_selected_outputs_and_all_owned_caches() {
    let prefix_tokens = 3;
    let checkpoint_layers = [0, 3];
    let config = stack_config(LAYERS);
    let input = one_token_input(HIDDEN);
    let final_norm_weight = final_norm_weight(HIDDEN);
    let prefix_caches = (0..LAYERS)
        .map(|layer_index| make_valid_cache(prefix_tokens, KEY_VALUE_HEADS, HEAD_DIM, layer_index))
        .collect::<Vec<_>>();
    let cache_len = (prefix_tokens + 1) * KEY_VALUE_HEADS * HEAD_DIM;
    let retained_payload_bytes =
        (checkpoint_layers.len() * HIDDEN + LAYERS * 2 * cache_len + HIDDEN) * size_of::<f32>();
    let retained_live_upper_bound = retained_payload_bytes + LIVENESS_ACCOUNTING_SLACK;
    assert!(LIVENESS_MARKER_BYTES > retained_live_upper_bound * 8);
    let mut expected_current = input.clone();
    let mut callback_live_bytes = Vec::with_capacity(LAYERS);

    begin_allocation_tracking();
    let trace = decoder_stack_decode_f32(
        &input,
        config,
        &prefix_caches,
        &checkpoint_layers,
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            let snapshot = allocation_snapshot();
            callback_live_bytes.push(snapshot.live_bytes);
            assert_f32_bits(
                "liveness current input",
                current_one_token,
                &expected_current,
            );
            let output = next_output(layer_index, current_one_token);
            expected_current.copy_from_slice(&output);
            Ok(trace_with_liveness_markers(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                output,
            ))
        },
    )
    .unwrap();
    let before_finish = allocation_snapshot();
    let snapshot = finish_allocation_tracking();
    assert_eq!(before_finish.live_bytes, snapshot.live_bytes);
    assert_eq!(before_finish.peak_bytes, snapshot.peak_bytes);
    assert!(!snapshot.overflowed);
    let expected_checkpoint_bytes = checkpoint_layers.len() * HIDDEN * size_of::<f32>();
    let expected_cache_bytes = trace
        .kv_caches
        .iter()
        .map(|cache| (cache.keys.len() + cache.values.len()) * size_of::<f32>())
        .sum::<usize>();
    let expected_final_norm_bytes = HIDDEN * size_of::<f32>();
    assert_eq!(trace.checkpoints.len(), checkpoint_layers.len());
    assert_eq!(trace.kv_caches.len(), LAYERS);
    assert_eq!(trace.executed_layers, LAYERS);
    assert_eq!(callback_live_bytes.len(), LAYERS);
    let callback_zero_live = callback_live_bytes[0];
    for (layer_index, live_bytes) in callback_live_bytes.into_iter().enumerate().skip(1) {
        assert!(
            live_bytes <= callback_zero_live + retained_live_upper_bound,
            "layer {layer_index} callback retained prior unselected stage buffers: \
             live={live_bytes}, first={callback_zero_live}, allowed_growth={retained_live_upper_bound}"
        );
    }
    assert_eq!(
        trace.retained_checkpoint_elements,
        checkpoint_layers.len() * HIDDEN
    );
    assert_eq!(
        trace.retained_kv_elements,
        trace
            .kv_caches
            .iter()
            .map(|cache| cache.keys.len() + cache.values.len())
            .sum::<usize>()
    );
    assert!(
        snapshot.live_bytes
            >= expected_checkpoint_bytes + expected_cache_bytes + expected_final_norm_bytes
    );
    assert!(
        snapshot.peak_bytes >= LIVENESS_MARKER_BYTES,
        "marker allocations were not observed: peak={} marker_bytes={LIVENESS_MARKER_BYTES}",
        snapshot.peak_bytes
    );
    assert!(
        snapshot.live_bytes <= callback_zero_live + retained_live_upper_bound,
        "decode stack retained full layer traces: live={} first={} allowed_growth={}",
        snapshot.live_bytes,
        callback_zero_live,
        retained_live_upper_bound
    );
}

#[test]
fn negative_controls_are_observable_and_returned_caches_are_detached() {
    let scenario = build_decode_scenario(3, LAYERS, &CHECKPOINTS);
    let correct = decode_plan_digest(&scenario, &[0, 1, 2, 3], false, false);
    let reset = decode_plan_digest(&scenario, &[0, 1, 2, 3], true, false);
    let repeat = decode_plan_digest(&scenario, &[0, 1, 1, 3], false, false);
    let reorder = decode_plan_digest(&scenario, &[0, 2, 1, 3], false, false);
    let wrong_cache_layer = decode_plan_digest(&scenario, &[0, 1, 2, 3], false, true);
    assert_ne!(reset, correct);
    assert_ne!(repeat, correct);
    assert_ne!(reorder, correct);
    assert_ne!(wrong_cache_layer, correct);

    let preserved_prefix = scenario.prefix_caches.clone();
    let mut actual = decoder_stack_decode_f32(
        &scenario.input,
        scenario.config,
        &scenario.prefix_caches,
        &CHECKPOINTS,
        &scenario.final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            Ok(valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            ))
        },
    )
    .unwrap();
    let preserved_actual = actual.kv_caches.clone();
    actual.kv_caches[1].keys[0] = 77.0;
    actual.kv_caches[1].values[1] = -88.0;
    assert_eq!(scenario.prefix_caches, preserved_prefix);
    assert_ne!(actual.kv_caches, preserved_actual);
}

#[test]
fn generic_decode_stack_without_checkpoints_retains_every_cache_and_reports_none_accessors() {
    let layers = 3;
    let prefix_tokens = 2;
    let config = stack_config(layers);
    let input = one_token_input(HIDDEN);
    let final_norm_weight = final_norm_weight(HIDDEN);
    let prefix_caches = (0..layers)
        .map(|layer_index| make_valid_cache(prefix_tokens, KEY_VALUE_HEADS, HEAD_DIM, layer_index))
        .collect::<Vec<_>>();
    let expected = synthetic_expected_decode_stack(&input, config, prefix_tokens, &[]);
    let actual = decoder_stack_decode_f32(
        &input,
        config,
        &prefix_caches,
        &[],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            Ok(valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            ))
        },
    )
    .unwrap();
    assert_stack_exact(&actual, &expected);
    assert_eq!(actual.checkpoints.len(), 0);
    assert_eq!(actual.retained_checkpoint_elements, 0);
    for layer_index in 0..layers {
        assert_eq!(actual.checkpoint(layer_index), None);
        assert_cache_exact(
            &format!("no-checkpoint cache {layer_index}"),
            actual.kv_cache(layer_index).unwrap(),
            &expected.kv_caches[layer_index],
        );
    }
    assert_eq!(actual.kv_cache(layers), None);
}

#[test]
fn pinned_decode_stack_wrapper_freezes_topology_and_rejects_malformed_prefix_count_before_callback()
{
    let input = one_token_input(PINNED_HIDDEN);
    let final_norm_weight = final_norm_weight(PINNED_HIDDEN);
    let prefix_tokens = 2;
    let prefix_caches = (0..PINNED_LAYERS)
        .map(|layer_index| {
            make_valid_cache(
                prefix_tokens,
                PINNED_KEY_VALUE_HEADS,
                PINNED_HEAD_DIM,
                layer_index,
            )
        })
        .collect::<Vec<_>>();
    let expected = synthetic_expected_decode_stack(
        &input,
        DecoderStackConfig {
            layer: pinned_layer_config(),
            layers: PINNED_LAYERS,
        },
        prefix_tokens,
        &[0, 17],
    );
    let mut seen = Vec::new();
    let trace = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &[0, 17],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            seen.push((
                layer_index,
                supplied_config,
                current_one_token.to_vec(),
                prefix_cache.tokens,
            ));
            Ok(valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            ))
        },
    )
    .unwrap();
    assert_eq!(seen.len(), PINNED_LAYERS);
    assert_stack_exact(&trace, &expected);
    assert_eq!(trace.executed_layers, PINNED_LAYERS);
    for (expected_layer_index, (layer_index, supplied_config, _, prefix_tokens_seen)) in
        seen.iter().enumerate()
    {
        assert_eq!(*layer_index, expected_layer_index);
        assert_eq!(*prefix_tokens_seen, prefix_tokens);
        assert_eq!(*supplied_config, pinned_layer_config());
    }
    assert_f32_bits(
        "pinned checkpoint 0 accessor",
        trace.checkpoint(0).unwrap(),
        &expected.checkpoints[0].values,
    );
    assert_f32_bits(
        "pinned checkpoint 17 accessor",
        trace.checkpoint(17).unwrap(),
        &expected.checkpoints[1].values,
    );
    assert_eq!(trace.checkpoint(1), None);
    assert_eq!(trace.kv_cache(PINNED_LAYERS), None);

    for (case, malformed_prefix_caches) in [
        ("pinned malformed prefix count short", {
            let mut caches = prefix_caches.clone();
            caches.pop();
            caches
        }),
        ("pinned malformed prefix count long", {
            let mut caches = prefix_caches.clone();
            caches.push(prefix_caches[0].clone());
            caches
        }),
    ] {
        assert_pinned_rejected_before_layer_zero(
            case,
            &input,
            &malformed_prefix_caches,
            &[0, 17],
            &final_norm_weight,
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    let mut pinned_wrong_kv_heads = prefix_caches.clone();
    pinned_wrong_kv_heads[3].key_value_heads += 1;
    assert_pinned_rejected_before_layer_zero(
        "pinned malformed prefix key_value_heads",
        &input,
        &pinned_wrong_kv_heads,
        &[0, 17],
        &final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );
    let mut pinned_wrong_head_dim = prefix_caches.clone();
    pinned_wrong_head_dim[5].head_dim += 2;
    assert_pinned_rejected_before_layer_zero(
        "pinned malformed prefix head_dim",
        &input,
        &pinned_wrong_head_dim,
        &[0, 17],
        &final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );
    let mut pinned_nonfinite_prefix_values = prefix_caches.clone();
    let last = pinned_nonfinite_prefix_values[4].values.len() - 1;
    pinned_nonfinite_prefix_values[4].values[last] = f32::NEG_INFINITY;
    assert_pinned_rejected_before_layer_zero(
        "pinned malformed prefix values nonfinite",
        &input,
        &pinned_nonfinite_prefix_values,
        &[0, 17],
        &final_norm_weight,
        CpuRefErrorCode::NonFiniteInput,
    );
    for checkpoint_layers in [vec![17, 0], vec![0, 0], vec![0, PINNED_LAYERS]] {
        assert_pinned_rejected_before_layer_zero(
            &format!("pinned invalid checkpoints {checkpoint_layers:?}"),
            &input,
            &prefix_caches,
            &checkpoint_layers,
            &final_norm_weight,
            CpuRefErrorCode::InvalidCheckpointSelection,
        );
    }
    for malformed_input in [
        &input[..input.len() - 1],
        &[input.clone(), vec![0.0]].concat()[..],
    ] {
        assert_pinned_rejected_before_layer_zero(
            "pinned malformed input length",
            malformed_input,
            &prefix_caches,
            &[0, 17],
            &final_norm_weight,
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for malformed_final_norm in [
        {
            let mut values = final_norm_weight.clone();
            values.pop();
            values
        },
        {
            let mut values = final_norm_weight.clone();
            values.push(0.0);
            values
        },
    ] {
        assert_pinned_rejected_before_layer_zero(
            "pinned malformed final norm length",
            &input,
            &prefix_caches,
            &[0, 17],
            &malformed_final_norm,
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    let mut nonfinite_input = input.clone();
    nonfinite_input[PINNED_HIDDEN / 2] = f32::NAN;
    assert_pinned_rejected_before_layer_zero(
        "pinned nonfinite input",
        &nonfinite_input,
        &prefix_caches,
        &[0, 17],
        &final_norm_weight,
        CpuRefErrorCode::NonFiniteInput,
    );
    let mut nonfinite_final_norm = final_norm_weight.clone();
    nonfinite_final_norm[PINNED_HIDDEN / 3] = f32::INFINITY;
    assert_pinned_rejected_before_layer_zero(
        "pinned nonfinite final norm",
        &input,
        &prefix_caches,
        &[0, 17],
        &nonfinite_final_norm,
        CpuRefErrorCode::NonFiniteInput,
    );
}

#[test]
fn pinned_decode_stack_without_checkpoints_retains_caches_and_reports_none_accessors() {
    let input = one_token_input(PINNED_HIDDEN);
    let final_norm_weight = final_norm_weight(PINNED_HIDDEN);
    let prefix_tokens = 2;
    let prefix_caches = (0..PINNED_LAYERS)
        .map(|layer_index| {
            make_valid_cache(
                prefix_tokens,
                PINNED_KEY_VALUE_HEADS,
                PINNED_HEAD_DIM,
                layer_index,
            )
        })
        .collect::<Vec<_>>();
    let expected = synthetic_expected_decode_stack(
        &input,
        DecoderStackConfig {
            layer: pinned_layer_config(),
            layers: PINNED_LAYERS,
        },
        prefix_tokens,
        &[],
    );
    let trace = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &[],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            Ok(valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            ))
        },
    )
    .unwrap();
    assert_stack_exact(&trace, &expected);
    assert_eq!(trace.checkpoints.len(), 0);
    assert_eq!(trace.retained_checkpoint_elements, 0);
    for layer_index in 0..PINNED_LAYERS {
        assert_eq!(trace.checkpoint(layer_index), None);
        assert_cache_exact(
            &format!("pinned no-checkpoint cache {layer_index}"),
            trace.kv_cache(layer_index).unwrap(),
            &expected.kv_caches[layer_index],
        );
    }
}

#[test]
fn pinned_decode_stack_validates_callback_results_and_propagates_callback_errors() {
    let input = one_token_input(PINNED_HIDDEN);
    let final_norm_weight = final_norm_weight(PINNED_HIDDEN);
    let prefix_tokens = 2;
    let prefix_caches = (0..PINNED_LAYERS)
        .map(|layer_index| {
            make_valid_cache(
                prefix_tokens,
                PINNED_KEY_VALUE_HEADS,
                PINNED_HEAD_DIM,
                layer_index,
            )
        })
        .collect::<Vec<_>>();

    let mut calls = Vec::new();
    let result = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &[0, 17],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            calls.push(layer_index);
            let mut trace = valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            );
            if layer_index == 2 {
                trace.output.pop();
            }
            Ok(trace)
        },
    );
    assert_eq!(calls, vec![0, 1, 2]);
    assert_error(
        "pinned malformed callback trace",
        result,
        CpuRefErrorCode::DimensionMismatch,
    );

    let mut cache_calls = Vec::new();
    let cache_result = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &[0, 17],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            cache_calls.push(layer_index);
            let mut trace = valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            );
            if layer_index == 2 {
                trace.kv_cache.values.pop();
            }
            Ok(trace)
        },
    );
    assert_eq!(cache_calls, vec![0, 1, 2]);
    assert_error(
        "pinned malformed callback cache",
        cache_result,
        CpuRefErrorCode::DimensionMismatch,
    );

    let mut error_calls = Vec::new();
    let error_result = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &[0, 17],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current_one_token: &[f32],
         prefix_cache: &DecoderPrefillKvCache|
         -> Result<DecoderLayerDecodeTrace, CpuRefError> {
            error_calls.push(layer_index);
            if layer_index == 1 {
                return Err(rms_norm_f32(&[f32::NAN], 1, 1, &[1.0], 1.0)
                    .expect_err("forced pinned callback error"));
            }
            Ok(valid_decode_trace(
                supplied_config,
                layer_index,
                prefix_cache.tokens,
                next_output(layer_index, current_one_token),
            ))
        },
    );
    assert_eq!(error_calls, vec![0, 1]);
    assert_error(
        "pinned propagated callback error",
        error_result,
        CpuRefErrorCode::NonFiniteInput,
    );
}
