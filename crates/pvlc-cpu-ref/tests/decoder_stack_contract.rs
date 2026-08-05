use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::UnsafeCell,
    mem::size_of,
};

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderLayerConfig, DecoderLayerParameters,
    DecoderLayerPrefillTrace, DecoderPrefillKvCache, DecoderStackCheckpoint, DecoderStackConfig,
    DecoderStackPrefillTrace, decoder_layer_prefill_f32, decoder_stack_prefill_f32,
    pinned_decoder_stack_prefill_f32, rms_norm_f32,
};

const TRACKED_ALLOCATION_SLOTS: usize = 4_096;

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
            // The pointer predates this tracking window, so its deallocation
            // must not reduce bytes allocated during the window.
            return;
        };
        self.live_bytes = self
            .live_bytes
            .checked_sub(self.sizes[slot])
            .expect("tracked allocation accounting underflowed");
        self.pointers[slot] = 0;
        self.sizes[slot] = 0;
    }

    fn snapshot(&self) -> AllocationSnapshot {
        AllocationSnapshot {
            live_bytes: self.live_bytes,
            peak_bytes: self.peak_bytes,
            overflowed: self.overflowed,
        }
    }

    fn finish(&mut self) -> AllocationSnapshot {
        assert!(self.enabled, "allocation tracking window is not active");
        let snapshot = self.snapshot();
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
        // SAFETY: the state is thread-local and every allocator callback uses
        // this synchronous helper, so no two mutable references coexist.
        operation(unsafe { &mut *state.get() })
    })
}

fn try_with_allocation_state(operation: impl FnOnce(&mut ThreadAllocationState)) {
    let _ = THREAD_ALLOCATION_STATE.try_with(|state| {
        // SAFETY: as above; `try_with` additionally avoids panicking if an
        // allocation occurs while this TLS key is being destroyed.
        operation(unsafe { &mut *state.get() });
    });
}

unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the exact layout to the system allocator.
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
        // SAFETY: `pointer` was allocated with this allocator and `layout` is
        // the matching layout supplied by the caller.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

fn begin_allocation_tracking() {
    with_allocation_state(ThreadAllocationState::begin);
}

fn allocation_snapshot() -> AllocationSnapshot {
    with_allocation_state(|state| state.snapshot())
}

fn finish_allocation_tracking() -> AllocationSnapshot {
    with_allocation_state(ThreadAllocationState::finish)
}

const TOKENS: usize = 3;
const HIDDEN: usize = 5;
const INTERMEDIATE: usize = 7;
const QUERY_HEADS: usize = 4;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 6;
const EPSILON: f32 = 1.0e-5;
const SECTIONS: [usize; 3] = [1, 1, 1];
const LAYERS: usize = 4;
const CHECKPOINTS: [usize; 3] = [0, 2, 3];

const PINNED_LAYERS: usize = 18;
const PINNED_HIDDEN: usize = 1_024;
const PINNED_INTERMEDIATE: usize = 3_072;
const PINNED_QUERY_HEADS: usize = 16;
const PINNED_KEY_VALUE_HEADS: usize = 2;
const PINNED_HEAD_DIM: usize = 128;
const PINNED_EPSILON: f32 = 1.0e-5;
const PINNED_SECTIONS: [usize; 3] = [16, 24, 24];

const LIVENESS_MARKER_FLOATS: usize = 16_384;
const LIVENESS_MARKER_VECTORS: usize = 4;
const LIVENESS_MARKER_BYTES: usize =
    LIVENESS_MARKER_FLOATS * LIVENESS_MARKER_VECTORS * size_of::<f32>();
const LIVENESS_ACCOUNTING_SLACK: usize = 16 * 1_024;

// Derived once before the stack API existed from `independent_stack`, which
// chains only the accepted public decoder-layer and RMSNorm primitives.
// The expected value is a literal and is never derived from the future stack
// result at runtime.
const COMBINED_BLAKE3: &str = "1b703cf4dc8d5a097c7d305d81b157b6753902c2828afaeb967c1e6462e3700a";

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
struct ExpectedCheckpoint {
    layer_index: usize,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct ExpectedStack {
    checkpoints: Vec<ExpectedCheckpoint>,
    kv_caches: Vec<DecoderPrefillKvCache>,
    final_norm: Vec<f32>,
    executed_layers: usize,
    retained_checkpoint_elements: usize,
    retained_kv_elements: usize,
}

#[derive(Clone, Debug)]
struct RealChain {
    layer_inputs: Vec<Vec<f32>>,
    layer_outputs: Vec<Vec<f32>>,
    kv_caches: Vec<DecoderPrefillKvCache>,
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
        layer: layer_config(TOKENS),
        layers,
    }
}

fn pinned_layer_config(tokens: usize) -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens,
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

fn tiny_input() -> Vec<f32> {
    dense(TOKENS * HIDDEN, 17, 3, 29, 11.0)
}

fn final_norm_weight(hidden_size: usize) -> Vec<f32> {
    (0..hidden_size)
        .map(|index| 0.73 + ((index * 7 + 2) % 13) as f32 / 31.0)
        .collect()
}

fn layer_parameters(layer_index: usize) -> OwnedParameters {
    let layer = layer_index + 1;
    OwnedParameters {
        input_norm_weight: (0..HIDDEN)
            .map(|index| 0.71 + ((index * 3 + layer * 2) % 11) as f32 / 37.0)
            .collect(),
        query_weight: dense(
            QUERY_HEADS * HEAD_DIM * HIDDEN,
            5 + layer * 2,
            3 + layer,
            31 + layer * 2,
            61.0 + layer as f32,
        ),
        key_weight: dense(
            KEY_VALUE_HEADS * HEAD_DIM * HIDDEN,
            7 + layer * 2,
            5 + layer,
            37 + layer * 2,
            67.0 + layer as f32,
        ),
        value_weight: dense(
            KEY_VALUE_HEADS * HEAD_DIM * HIDDEN,
            11 + layer * 2,
            7 + layer,
            41 + layer * 2,
            71.0 + layer as f32,
        ),
        attention_output_weight: dense(
            HIDDEN * QUERY_HEADS * HEAD_DIM,
            13 + layer * 2,
            2 + layer,
            43 + layer * 2,
            79.0 + layer as f32,
        ),
        post_attention_norm_weight: (0..HIDDEN)
            .map(|index| 0.67 + ((index * 5 + layer * 3) % 13) as f32 / 41.0)
            .collect(),
        gate_weight: dense(
            INTERMEDIATE * HIDDEN,
            17 + layer * 2,
            1 + layer,
            47 + layer * 2,
            83.0 + layer as f32,
        ),
        up_weight: dense(
            INTERMEDIATE * HIDDEN,
            19 + layer * 2,
            4 + layer,
            53 + layer * 2,
            89.0 + layer as f32,
        ),
        down_weight: dense(
            HIDDEN * INTERMEDIATE,
            23 + layer * 2,
            6 + layer,
            59 + layer * 2,
            97.0 + layer as f32,
        ),
    }
}

fn layer_tables(layer_index: usize, tokens: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = Vec::with_capacity(3 * tokens * HEAD_DIM);
    let mut sine = Vec::with_capacity(3 * tokens * HEAD_DIM);
    for axis in 0..3 {
        for token in 0..tokens {
            for dimension in 0..HEAD_DIM {
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

fn execute_real_chain(
    input: &[f32],
    execution_order: &[usize],
    reset_to_original: bool,
    config: DecoderLayerConfig,
    tables: &[(Vec<f32>, Vec<f32>)],
    parameters: &[OwnedParameters],
) -> RealChain {
    let mut current = input.to_vec();
    let mut layer_inputs = Vec::with_capacity(execution_order.len());
    let mut layer_outputs = Vec::with_capacity(execution_order.len());
    let mut kv_caches = Vec::with_capacity(execution_order.len());
    for &layer_index in execution_order {
        let source = if reset_to_original { input } else { &current };
        layer_inputs.push(source.to_vec());
        let trace = decoder_layer_prefill_f32(
            source,
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
    RealChain {
        layer_inputs,
        layer_outputs,
        kv_caches,
    }
}

fn independent_stack(
    input: &[f32],
    config: DecoderStackConfig,
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    tables: &[(Vec<f32>, Vec<f32>)],
    parameters: &[OwnedParameters],
) -> (ExpectedStack, RealChain) {
    let order = (0..config.layers).collect::<Vec<_>>();
    let chain = execute_real_chain(input, &order, false, config.layer, tables, parameters);
    let checkpoints = checkpoint_layers
        .iter()
        .map(|&layer_index| ExpectedCheckpoint {
            layer_index,
            values: chain.layer_outputs[layer_index].clone(),
        })
        .collect::<Vec<_>>();
    let final_norm = rms_norm_f32(
        chain.layer_outputs.last().unwrap(),
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
    let retained_kv_elements = chain
        .kv_caches
        .iter()
        .map(|cache| cache.keys.len() + cache.values.len())
        .sum();
    (
        ExpectedStack {
            checkpoints,
            kv_caches: chain.kv_caches.clone(),
            final_norm,
            executed_layers: config.layers,
            retained_checkpoint_elements,
            retained_kv_elements,
        },
        chain,
    )
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
        "{label} KV heads"
    );
    assert_eq!(actual.head_dim, expected.head_dim, "{label} head dim");
    assert_f32_bits(&format!("{label} keys"), &actual.keys, &expected.keys);
    assert_f32_bits(&format!("{label} values"), &actual.values, &expected.values);
}

fn assert_stack_exact(actual: &DecoderStackPrefillTrace, expected: &ExpectedStack) {
    assert_eq!(actual.checkpoints.len(), expected.checkpoints.len());
    for (index, expected_checkpoint) in expected.checkpoints.iter().enumerate() {
        let actual_checkpoint: &DecoderStackCheckpoint = &actual.checkpoints[index];
        assert_eq!(
            actual_checkpoint.layer_index,
            expected_checkpoint.layer_index
        );
        assert_f32_bits(
            &format!("checkpoint {}", expected_checkpoint.layer_index),
            &actual_checkpoint.values,
            &expected_checkpoint.values,
        );
    }
    assert_eq!(actual.kv_caches.len(), expected.kv_caches.len());
    for (layer_index, (actual_cache, expected_cache)) in
        actual.kv_caches.iter().zip(&expected.kv_caches).enumerate()
    {
        assert_cache_exact(
            &format!("layer {layer_index} cache"),
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

trait StackDigestView {
    fn checkpoint_count(&self) -> usize;
    fn checkpoint_at(&self, index: usize) -> (usize, &[f32]);
    fn caches(&self) -> &[DecoderPrefillKvCache];
    fn final_norm(&self) -> &[f32];
    fn executed_layers(&self) -> usize;
    fn retained_checkpoint_elements(&self) -> usize;
    fn retained_kv_elements(&self) -> usize;
}

impl StackDigestView for ExpectedStack {
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

impl StackDigestView for DecoderStackPrefillTrace {
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

fn update_usize_digest(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}

fn update_f32_digest(hasher: &mut blake3::Hasher, values: &[f32]) {
    update_usize_digest(hasher, values.len());
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

// Fixed order: checkpoint count; each checkpoint layer/length/f32 LE values;
// cache count; each cache tokens/KV-heads/head-dim, key length/data, value
// length/data; final norm length/data; executed layer count; retained
// checkpoint elements; retained KV elements. Every usize is encoded as u64 LE.
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

fn tensor_digest(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    update_f32_digest(&mut hasher, values);
    hasher.finalize().to_hex().to_string()
}

fn assert_digest_differs(label: &str, wrong: &[f32], correct: &[f32]) {
    assert_ne!(tensor_digest(wrong), tensor_digest(correct), "{label}");
}

fn assert_error<T>(case: &str, result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{case}"),
        Ok(_) => panic!("{case}: expected {expected:?}"),
    }
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

fn valid_minimal_trace(
    config: DecoderLayerConfig,
    layer_index: usize,
    output: Vec<f32>,
) -> DecoderLayerPrefillTrace {
    let activation_len = config.tokens.checked_mul(config.hidden_size).unwrap();
    let query_width = config.query_heads.checked_mul(config.head_dim).unwrap();
    let query_len = config.tokens.checked_mul(query_width).unwrap();
    let key_value_width = config.key_value_heads.checked_mul(config.head_dim).unwrap();
    let key_value_len = config.tokens.checked_mul(key_value_width).unwrap();
    let intermediate_len = config.tokens.checked_mul(config.intermediate_size).unwrap();
    assert_eq!(output.len(), activation_len);
    DecoderLayerPrefillTrace {
        norm1: stage_values(activation_len, layer_index, 0),
        query: stage_values(query_len, layer_index, 1),
        key: stage_values(key_value_len, layer_index, 2),
        value: stage_values(key_value_len, layer_index, 3),
        mrope_query: stage_values(query_len, layer_index, 4),
        mrope_key: stage_values(key_value_len, layer_index, 5),
        kv_cache: DecoderPrefillKvCache {
            keys: stage_values(key_value_len, layer_index, 6),
            values: stage_values(key_value_len, layer_index, 7),
            tokens: config.tokens,
            key_value_heads: config.key_value_heads,
            head_dim: config.head_dim,
        },
        attention_context: stage_values(query_len, layer_index, 8),
        attention_output: stage_values(activation_len, layer_index, 9),
        attention_residual: stage_values(activation_len, layer_index, 10),
        norm2: stage_values(activation_len, layer_index, 11),
        mlp_gate: stage_values(intermediate_len, layer_index, 12),
        mlp_up: stage_values(intermediate_len, layer_index, 13),
        mlp_activation: stage_values(intermediate_len, layer_index, 14),
        mlp_down: stage_values(activation_len, layer_index, 15),
        output,
    }
}

fn trace_with_liveness_markers(
    config: DecoderLayerConfig,
    layer_index: usize,
    output: Vec<f32>,
) -> DecoderLayerPrefillTrace {
    let mut trace = valid_minimal_trace(config, layer_index, output);
    let marker = layer_index as f32 + 0.25;
    trace.norm1 = vec![marker; LIVENESS_MARKER_FLOATS];
    trace.query = vec![marker + 0.1; LIVENESS_MARKER_FLOATS];
    trace.attention_context = vec![marker + 0.2; LIVENESS_MARKER_FLOATS];
    trace.mlp_activation = vec![marker + 0.3; LIVENESS_MARKER_FLOATS];
    trace
}

fn assert_rejected_before_layer_zero(
    case: &str,
    input: &[f32],
    config: DecoderStackConfig,
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    expected: CpuRefErrorCode,
) {
    let mut calls = 0_usize;
    let result = decoder_stack_prefill_f32(
        input,
        config,
        checkpoint_layers,
        final_norm_weight,
        |_, _, _| -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            calls += 1;
            panic!("{case}: invalid stack invoked layer zero")
        },
    );
    assert_eq!(calls, 0, "{case}");
    assert_error(case, result, expected);
}

fn assert_pinned_rejected_before_layer_zero(
    case: &str,
    input: &[f32],
    tokens: usize,
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    expected: CpuRefErrorCode,
) {
    let mut calls = 0_usize;
    let result = pinned_decoder_stack_prefill_f32(
        input,
        tokens,
        checkpoint_layers,
        final_norm_weight,
        |_, _, _| -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            calls += 1;
            panic!("{case}: invalid pinned stack invoked layer zero")
        },
    );
    assert_eq!(calls, 0, "{case}");
    assert_error(case, result, expected);
}

#[test]
fn real_tiny_stack_matches_independent_chain_literal_anchor_and_preserves_operands() {
    let config = stack_config(LAYERS);
    let input = tiny_input();
    let parameters = (0..LAYERS).map(layer_parameters).collect::<Vec<_>>();
    let tables = (0..LAYERS)
        .map(|layer_index| layer_tables(layer_index, TOKENS))
        .collect::<Vec<_>>();
    let final_norm_weight = final_norm_weight(HIDDEN);
    let preserved = (
        input.clone(),
        parameters.clone(),
        tables.clone(),
        final_norm_weight.clone(),
    );
    let (expected, expected_chain) = independent_stack(
        &input,
        config,
        &CHECKPOINTS,
        &final_norm_weight,
        &tables,
        &parameters,
    );
    assert_eq!(combined_digest(&expected), COMBINED_BLAKE3);

    let mut invocation_order = Vec::new();
    let mut observed_inputs = Vec::new();
    let actual = decoder_stack_prefill_f32(
        &input,
        config,
        &CHECKPOINTS,
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current: &[f32]|
         -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            assert_eq!(supplied_config, config.layer);
            invocation_order.push(layer_index);
            observed_inputs.push(current.to_vec());
            decoder_layer_prefill_f32(
                current,
                supplied_config,
                &tables[layer_index].0,
                &tables[layer_index].1,
                parameters[layer_index].borrowed(),
            )
        },
    )
    .unwrap();

    assert_eq!(invocation_order, [0, 1, 2, 3]);
    assert_eq!(observed_inputs.len(), expected_chain.layer_inputs.len());
    for (layer_index, (actual_input, expected_input)) in observed_inputs
        .iter()
        .zip(&expected_chain.layer_inputs)
        .enumerate()
    {
        assert_f32_bits(
            &format!("layer {layer_index} current input"),
            actual_input,
            expected_input,
        );
    }
    assert_stack_exact(&actual, &expected);
    assert_eq!(combined_digest(&actual), COMBINED_BLAKE3);
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
    assert_eq!(input, preserved.0);
    assert_eq!(parameters, preserved.1);
    assert_eq!(tables, preserved.2);
    assert_eq!(final_norm_weight, preserved.3);
}

#[test]
fn generic_one_layer_stack_succeeds_and_applies_final_norm() {
    let config = stack_config(1);
    let input = tiny_input();
    let parameters = layer_parameters(0);
    let tables = layer_tables(0, TOKENS);
    let final_norm_weight = final_norm_weight(HIDDEN);
    let expected_layer = decoder_layer_prefill_f32(
        &input,
        config.layer,
        &tables.0,
        &tables.1,
        parameters.borrowed(),
    )
    .unwrap();
    let expected_final = rms_norm_f32(
        &expected_layer.output,
        TOKENS,
        HIDDEN,
        &final_norm_weight,
        EPSILON,
    )
    .unwrap();
    let mut calls = 0_usize;

    let trace = decoder_stack_prefill_f32(
        &input,
        config,
        &[0],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current: &[f32]|
         -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            assert_eq!(layer_index, 0);
            assert_eq!(supplied_config, config.layer);
            assert_f32_bits("one-layer current input", current, &input);
            calls += 1;
            decoder_layer_prefill_f32(
                current,
                supplied_config,
                &tables.0,
                &tables.1,
                parameters.borrowed(),
            )
        },
    )
    .unwrap();

    assert_eq!(calls, 1);
    assert_eq!(trace.executed_layers, 1);
    assert_eq!(trace.checkpoints.len(), 1);
    assert_eq!(trace.kv_caches.len(), 1);
    assert_f32_bits(
        "one-layer checkpoint",
        trace.checkpoint(0).unwrap(),
        &expected_layer.output,
    );
    assert_cache_exact(
        "one-layer cache",
        trace.kv_cache(0).unwrap(),
        &expected_layer.kv_cache,
    );
    assert_f32_bits("one-layer final norm", &trace.final_norm, &expected_final);
    assert_eq!(trace.retained_checkpoint_elements, TOKENS * HIDDEN);
    assert_eq!(
        trace.retained_kv_elements,
        2 * TOKENS * KEY_VALUE_HEADS * HEAD_DIM
    );
}

#[test]
fn generic_stack_without_checkpoints_retains_every_kv_and_applies_final_norm() {
    let layers = 3;
    let config = stack_config(layers);
    let input = tiny_input();
    let final_norm_weight = final_norm_weight(HIDDEN);
    let preserved = (input.clone(), final_norm_weight.clone());
    let cache_len = TOKENS * KEY_VALUE_HEADS * HEAD_DIM;
    let mut expected_current = input.clone();
    let mut expected_caches = Vec::with_capacity(layers);
    let mut calls = 0_usize;

    let trace = decoder_stack_prefill_f32(
        &input,
        config,
        &[],
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current: &[f32]|
         -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            assert_eq!(layer_index, calls);
            assert_eq!(supplied_config, config.layer);
            assert_f32_bits(
                "checkpoint-free prior layer output",
                current,
                &expected_current,
            );
            calls += 1;
            let output = next_output(layer_index, current);
            let layer_trace = valid_minimal_trace(supplied_config, layer_index, output.clone());
            expected_current = output;
            expected_caches.push(layer_trace.kv_cache.clone());
            Ok(layer_trace)
        },
    )
    .unwrap();

    assert_eq!(calls, layers);
    assert_eq!(trace.executed_layers, layers);
    assert!(trace.checkpoints.is_empty());
    assert_eq!(trace.retained_checkpoint_elements, 0);
    assert_eq!(trace.checkpoint(0), None);
    assert_eq!(trace.checkpoint(layers - 1), None);
    assert_eq!(trace.checkpoint(layers), None);
    assert_eq!(trace.kv_caches.len(), layers);
    assert_eq!(trace.retained_kv_elements, layers * 2 * cache_len);
    for (layer_index, expected_cache) in expected_caches.iter().enumerate() {
        assert_cache_exact(
            &format!("checkpoint-free cache {layer_index}"),
            trace.kv_cache(layer_index).unwrap(),
            expected_cache,
        );
    }
    assert_eq!(trace.kv_cache(layers), None);
    let expected_final = rms_norm_f32(
        &expected_current,
        TOKENS,
        HIDDEN,
        &final_norm_weight,
        EPSILON,
    )
    .unwrap();
    assert_f32_bits(
        "checkpoint-free final norm",
        &trace.final_norm,
        &expected_final,
    );
    assert_eq!(input, preserved.0);
    assert_eq!(final_norm_weight, preserved.1);
}

#[test]
fn reset_skip_repeat_reorder_and_final_norm_negative_controls_diverge() {
    let config = stack_config(LAYERS);
    let input = tiny_input();
    let parameters = (0..LAYERS).map(layer_parameters).collect::<Vec<_>>();
    let tables = (0..LAYERS)
        .map(|layer_index| layer_tables(layer_index, TOKENS))
        .collect::<Vec<_>>();
    let final_norm_weight = final_norm_weight(HIDDEN);
    let correct_chain = execute_real_chain(
        &input,
        &[0, 1, 2, 3],
        false,
        config.layer,
        &tables,
        &parameters,
    );
    let correct_current = correct_chain.layer_outputs.last().unwrap();
    let correct_final =
        rms_norm_f32(correct_current, TOKENS, HIDDEN, &final_norm_weight, EPSILON).unwrap();

    for (label, order, reset) in [
        (
            "reset every layer to original input",
            &[0, 1, 2, 3][..],
            true,
        ),
        ("skipped layer", &[0, 1, 3][..], false),
        ("repeated layer", &[0, 1, 1, 2, 3][..], false),
        ("reordered layers", &[0, 2, 1, 3][..], false),
    ] {
        let wrong_chain =
            execute_real_chain(&input, order, reset, config.layer, &tables, &parameters);
        let wrong_final = rms_norm_f32(
            wrong_chain.layer_outputs.last().unwrap(),
            TOKENS,
            HIDDEN,
            &final_norm_weight,
            EPSILON,
        )
        .unwrap();
        assert_digest_differs(label, &wrong_final, &correct_final);
    }

    assert_digest_differs("skipped final RMSNorm", correct_current, &correct_final);
    let mut wrong_final_norm_weight = final_norm_weight.clone();
    wrong_final_norm_weight.rotate_left(1);
    let wrong_final_norm = rms_norm_f32(
        correct_current,
        TOKENS,
        HIDDEN,
        &wrong_final_norm_weight,
        EPSILON,
    )
    .unwrap();
    assert_digest_differs(
        "wrong final RMSNorm weight",
        &wrong_final_norm,
        &correct_final,
    );
}

#[test]
fn stack_prevalidation_rejects_all_invalid_inputs_before_layer_zero() {
    let base = stack_config(LAYERS);
    let input = tiny_input();
    let final_norm_weight = final_norm_weight(HIDDEN);

    let mut invalid_configs: Vec<(&str, DecoderStackConfig, CpuRefErrorCode)> = Vec::new();
    let mut zero_layers = base;
    zero_layers.layers = 0;
    invalid_configs.push((
        "zero layers",
        zero_layers,
        CpuRefErrorCode::DimensionMismatch,
    ));
    for field in 0..6 {
        let mut config = base;
        let label = match field {
            0 => {
                config.layer.tokens = 0;
                "zero tokens"
            }
            1 => {
                config.layer.hidden_size = 0;
                "zero hidden size"
            }
            2 => {
                config.layer.intermediate_size = 0;
                "zero intermediate size"
            }
            3 => {
                config.layer.query_heads = 0;
                "zero query heads"
            }
            4 => {
                config.layer.key_value_heads = 0;
                "zero KV heads"
            }
            5 => {
                config.layer.head_dim = 0;
                "zero head dimension"
            }
            _ => unreachable!(),
        };
        invalid_configs.push((label, config, CpuRefErrorCode::DimensionMismatch));
    }
    let mut invalid_gqa = base;
    invalid_gqa.layer.query_heads = 3;
    invalid_configs.push((
        "query heads not divisible by KV heads",
        invalid_gqa,
        CpuRefErrorCode::DimensionMismatch,
    ));
    for (label, sections) in [
        ("zero M-RoPE section", [0, 1, 2]),
        ("wrong M-RoPE section sum", [1, 1, 2]),
        ("M-RoPE section sum overflow", [usize::MAX, 1, 1]),
        ("M-RoPE doubled section overflow", [usize::MAX / 2, 1, 1]),
    ] {
        let mut config = base;
        config.layer.mrope_sections = sections;
        invalid_configs.push((label, config, CpuRefErrorCode::DimensionMismatch));
    }
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut config = base;
        config.layer.rms_norm_epsilon = epsilon;
        invalid_configs.push((
            "invalid RMSNorm epsilon",
            config,
            CpuRefErrorCode::NonPositiveEpsilon,
        ));
    }
    let mut token_overflow = base;
    token_overflow.layer.tokens = usize::MAX;
    invalid_configs.push((
        "token-derived length overflow",
        token_overflow,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut hidden_overflow = base;
    hidden_overflow.layer.hidden_size = usize::MAX;
    invalid_configs.push((
        "hidden-derived length overflow",
        hidden_overflow,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut intermediate_overflow = base;
    intermediate_overflow.layer.intermediate_size = usize::MAX;
    invalid_configs.push((
        "intermediate-derived length overflow",
        intermediate_overflow,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut query_width_overflow = base;
    query_width_overflow.layer.query_heads = usize::MAX - 1;
    invalid_configs.push((
        "query width overflow",
        query_width_overflow,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut key_value_width_overflow = base;
    key_value_width_overflow.layer.query_heads = usize::MAX;
    key_value_width_overflow.layer.key_value_heads = usize::MAX;
    invalid_configs.push((
        "KV width overflow",
        key_value_width_overflow,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut retained_kv_overflow = base;
    retained_kv_overflow.layers = usize::MAX;
    invalid_configs.push((
        "retained KV element counter overflow",
        retained_kv_overflow,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut retained_checkpoint_overflow = base;
    retained_checkpoint_overflow.layer.tokens = 1;
    retained_checkpoint_overflow.layer.hidden_size = usize::MAX / 2 + 1;
    invalid_configs.push((
        "retained checkpoint element counter overflow",
        retained_checkpoint_overflow,
        CpuRefErrorCode::DimensionMismatch,
    ));

    for (case, config, expected) in invalid_configs {
        let checkpoint_layers = if case == "retained checkpoint element counter overflow" {
            &[0, 1][..]
        } else {
            &[][..]
        };
        assert_rejected_before_layer_zero(
            case,
            &input,
            config,
            checkpoint_layers,
            &final_norm_weight,
            expected,
        );
    }

    for long in [false, true] {
        let mut malformed_input = input.clone();
        if long {
            malformed_input.push(0.0);
        } else {
            malformed_input.pop();
        }
        assert_rejected_before_layer_zero(
            if long { "long input" } else { "short input" },
            &malformed_input,
            base,
            &[],
            &final_norm_weight,
            CpuRefErrorCode::DimensionMismatch,
        );
        let mut malformed_weight = final_norm_weight.clone();
        if long {
            malformed_weight.push(1.0);
        } else {
            malformed_weight.pop();
        }
        assert_rejected_before_layer_zero(
            if long {
                "long final norm weight"
            } else {
                "short final norm weight"
            },
            &input,
            base,
            &[],
            &malformed_weight,
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for offset in [0, input.len() / 2, input.len() - 1] {
            let mut malformed_input = input.clone();
            malformed_input[offset] = value;
            assert_rejected_before_layer_zero(
                &format!("non-finite input {value:?} at {offset}"),
                &malformed_input,
                base,
                &[],
                &final_norm_weight,
                CpuRefErrorCode::NonFiniteInput,
            );
        }
        for offset in [0, final_norm_weight.len() / 2, final_norm_weight.len() - 1] {
            let mut malformed_weight = final_norm_weight.clone();
            malformed_weight[offset] = value;
            assert_rejected_before_layer_zero(
                &format!("non-finite final norm weight {value:?} at {offset}"),
                &input,
                base,
                &[],
                &malformed_weight,
                CpuRefErrorCode::NonFiniteInput,
            );
        }
    }

    for (case, checkpoints) in [
        ("duplicate checkpoints", &[1, 1][..]),
        ("reordered checkpoints", &[2, 1][..]),
        ("out-of-range checkpoint", &[0, LAYERS][..]),
    ] {
        assert_rejected_before_layer_zero(
            case,
            &input,
            base,
            checkpoints,
            &final_norm_weight,
            CpuRefErrorCode::InvalidCheckpointSelection,
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum ReturnedFault {
    OutputShort,
    OutputLong,
    OutputNonFinite(f32),
    CacheKeysShort,
    CacheKeysLong,
    CacheKeysNonFinite(f32),
    CacheValuesShort,
    CacheValuesLong,
    CacheValuesNonFinite(f32),
    WrongTokens,
    WrongKeyValueHeads,
    WrongHeadDim,
}

impl ReturnedFault {
    fn label(self) -> String {
        match self {
            Self::OutputShort => "short returned output".to_owned(),
            Self::OutputLong => "long returned output".to_owned(),
            Self::OutputNonFinite(value) => format!("returned output contains {value:?}"),
            Self::CacheKeysShort => "short returned cache keys".to_owned(),
            Self::CacheKeysLong => "long returned cache keys".to_owned(),
            Self::CacheKeysNonFinite(value) => {
                format!("returned cache keys contain {value:?}")
            }
            Self::CacheValuesShort => "short returned cache values".to_owned(),
            Self::CacheValuesLong => "long returned cache values".to_owned(),
            Self::CacheValuesNonFinite(value) => {
                format!("returned cache values contain {value:?}")
            }
            Self::WrongTokens => "returned cache has wrong token metadata".to_owned(),
            Self::WrongKeyValueHeads => "returned cache has wrong KV-head metadata".to_owned(),
            Self::WrongHeadDim => "returned cache has wrong head-dim metadata".to_owned(),
        }
    }

    const fn expected_code(self) -> CpuRefErrorCode {
        match self {
            Self::OutputNonFinite(_)
            | Self::CacheKeysNonFinite(_)
            | Self::CacheValuesNonFinite(_) => CpuRefErrorCode::NonFiniteInput,
            Self::OutputShort
            | Self::OutputLong
            | Self::CacheKeysShort
            | Self::CacheKeysLong
            | Self::CacheValuesShort
            | Self::CacheValuesLong
            | Self::WrongTokens
            | Self::WrongKeyValueHeads
            | Self::WrongHeadDim => CpuRefErrorCode::DimensionMismatch,
        }
    }

    fn apply(self, trace: &mut DecoderLayerPrefillTrace, config: DecoderLayerConfig) {
        match self {
            Self::OutputShort => {
                trace.output.pop();
            }
            Self::OutputLong => trace.output.push(0.0),
            Self::OutputNonFinite(value) => {
                let middle = trace.output.len() / 2;
                trace.output[middle] = value;
            }
            Self::CacheKeysShort => {
                trace.kv_cache.keys.pop();
            }
            Self::CacheKeysLong => trace.kv_cache.keys.push(0.0),
            Self::CacheKeysNonFinite(value) => {
                let middle = trace.kv_cache.keys.len() / 2;
                trace.kv_cache.keys[middle] = value;
            }
            Self::CacheValuesShort => {
                trace.kv_cache.values.pop();
            }
            Self::CacheValuesLong => trace.kv_cache.values.push(0.0),
            Self::CacheValuesNonFinite(value) => {
                let middle = trace.kv_cache.values.len() / 2;
                trace.kv_cache.values[middle] = value;
            }
            Self::WrongTokens => trace.kv_cache.tokens = config.tokens + 1,
            Self::WrongKeyValueHeads => {
                trace.kv_cache.key_value_heads = config.key_value_heads + 1;
            }
            Self::WrongHeadDim => trace.kv_cache.head_dim = config.head_dim + 1,
        }
    }
}

#[test]
fn malformed_layer_outputs_and_caches_fail_at_the_producing_layer_and_errors_propagate() {
    let config = stack_config(LAYERS);
    let input = tiny_input();
    let final_norm_weight = final_norm_weight(HIDDEN);
    let faults = [
        ReturnedFault::OutputShort,
        ReturnedFault::OutputLong,
        ReturnedFault::OutputNonFinite(f32::NAN),
        ReturnedFault::OutputNonFinite(f32::INFINITY),
        ReturnedFault::OutputNonFinite(f32::NEG_INFINITY),
        ReturnedFault::CacheKeysShort,
        ReturnedFault::CacheKeysLong,
        ReturnedFault::CacheKeysNonFinite(f32::NAN),
        ReturnedFault::CacheKeysNonFinite(f32::INFINITY),
        ReturnedFault::CacheKeysNonFinite(f32::NEG_INFINITY),
        ReturnedFault::CacheValuesShort,
        ReturnedFault::CacheValuesLong,
        ReturnedFault::CacheValuesNonFinite(f32::NAN),
        ReturnedFault::CacheValuesNonFinite(f32::INFINITY),
        ReturnedFault::CacheValuesNonFinite(f32::NEG_INFINITY),
        ReturnedFault::WrongTokens,
        ReturnedFault::WrongKeyValueHeads,
        ReturnedFault::WrongHeadDim,
    ];

    let injected = rms_norm_f32(&[1.0], 1, 1, &[1.0], 0.0).unwrap_err();
    let expected_error = injected.clone();
    for producer in [0, LAYERS / 2, LAYERS - 1] {
        let expected_calls = (0..=producer).collect::<Vec<_>>();
        for fault in faults {
            let case = format!("{} at layer {producer}", fault.label());
            let mut calls = Vec::new();
            let result = decoder_stack_prefill_f32(
                &input,
                config,
                &[0, 3],
                &final_norm_weight,
                |layer_index: usize,
                 supplied_config: DecoderLayerConfig,
                 current: &[f32]|
                 -> Result<DecoderLayerPrefillTrace, CpuRefError> {
                    assert_eq!(supplied_config, config.layer);
                    calls.push(layer_index);
                    let mut trace = valid_minimal_trace(
                        supplied_config,
                        layer_index,
                        next_output(layer_index, current),
                    );
                    if layer_index == producer {
                        fault.apply(&mut trace, supplied_config);
                    }
                    Ok(trace)
                },
            );
            assert_error(&case, result, fault.expected_code());
            assert_eq!(calls, expected_calls, "{case}: a later layer was invoked");
        }

        let mut calls = Vec::new();
        let error = decoder_stack_prefill_f32(
            &input,
            config,
            &[],
            &final_norm_weight,
            |layer_index: usize,
             supplied_config: DecoderLayerConfig,
             current: &[f32]|
             -> Result<DecoderLayerPrefillTrace, CpuRefError> {
                assert_eq!(supplied_config, config.layer);
                calls.push(layer_index);
                if layer_index == producer {
                    Err(injected.clone())
                } else {
                    Ok(valid_minimal_trace(
                        supplied_config,
                        layer_index,
                        next_output(layer_index, current),
                    ))
                }
            },
        )
        .unwrap_err();
        assert_eq!(error, expected_error, "executor error at layer {producer}");
        assert_eq!(
            calls, expected_calls,
            "executor error at layer {producer} invoked a later layer"
        );
    }
}

#[test]
fn unselected_layer_stage_allocations_are_dead_before_the_next_callback() {
    let layers = 6;
    let checkpoints = [0, layers - 1];
    let config = stack_config(layers);
    let input = tiny_input();
    let final_norm_weight = final_norm_weight(HIDDEN);
    let activation_len = TOKENS * HIDDEN;
    let cache_len = TOKENS * KEY_VALUE_HEADS * HEAD_DIM;
    let retained_payload_bytes =
        (checkpoints.len() * activation_len + layers * 2 * cache_len + activation_len)
            * size_of::<f32>();
    let retained_live_upper_bound = retained_payload_bytes + LIVENESS_ACCOUNTING_SLACK;
    assert!(LIVENESS_MARKER_BYTES > retained_live_upper_bound * 8);

    let mut expected_current = input.clone();
    let mut callback_live_bytes = Vec::with_capacity(layers);
    let mut calls = 0_usize;
    begin_allocation_tracking();
    let result = decoder_stack_prefill_f32(
        &input,
        config,
        &checkpoints,
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current: &[f32]|
         -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            let snapshot = allocation_snapshot();
            callback_live_bytes.push(snapshot.live_bytes);
            assert_eq!(supplied_config, config.layer);
            assert_eq!(layer_index, calls);
            assert_f32_bits("liveness current input", current, &expected_current);
            calls += 1;
            let output = next_output(layer_index, current);
            expected_current.copy_from_slice(&output);
            Ok(trace_with_liveness_markers(
                supplied_config,
                layer_index,
                output,
            ))
        },
    );
    let before_finish = allocation_snapshot();
    let final_snapshot = finish_allocation_tracking();
    let trace = result.unwrap();

    assert_eq!(before_finish.live_bytes, final_snapshot.live_bytes);
    assert_eq!(before_finish.peak_bytes, final_snapshot.peak_bytes);
    assert!(
        !final_snapshot.overflowed,
        "allocation tracker slots overflowed"
    );
    assert_eq!(calls, layers);
    assert_eq!(callback_live_bytes.len(), layers);
    let callback_zero_live = callback_live_bytes[0];
    for (layer_index, live_bytes) in callback_live_bytes.into_iter().enumerate().skip(1) {
        assert!(
            live_bytes <= callback_zero_live + retained_live_upper_bound,
            "layer {layer_index} callback retained prior unselected stage buffers: \
             live={live_bytes}, first={callback_zero_live}, allowed_growth={retained_live_upper_bound}"
        );
    }
    assert!(
        final_snapshot.peak_bytes >= LIVENESS_MARKER_BYTES,
        "marker allocations were not observed: peak={} marker_bytes={LIVENESS_MARKER_BYTES}",
        final_snapshot.peak_bytes
    );
    assert!(
        final_snapshot.live_bytes <= callback_zero_live + retained_live_upper_bound,
        "stack return retained full layer traces: live={} first={} allowed_growth={}",
        final_snapshot.live_bytes,
        callback_zero_live,
        retained_live_upper_bound
    );
    assert_eq!(trace.executed_layers, layers);
    assert_eq!(trace.checkpoints.len(), checkpoints.len());
    assert_eq!(trace.kv_caches.len(), layers);
    assert_eq!(
        trace.retained_checkpoint_elements,
        checkpoints.len() * activation_len
    );
    assert_eq!(trace.retained_kv_elements, layers * 2 * cache_len);
    let expected_final = rms_norm_f32(
        &expected_current,
        TOKENS,
        HIDDEN,
        &final_norm_weight,
        EPSILON,
    )
    .unwrap();
    assert_f32_bits("liveness final norm", &trace.final_norm, &expected_final);
}

#[test]
fn sixty_four_layer_retention_is_bounded_to_selected_outputs_and_all_kv_caches() {
    let layers = 64;
    let checkpoints = [0, 31, 63];
    let config = stack_config(layers);
    let input = tiny_input();
    let final_norm_weight = final_norm_weight(HIDDEN);
    let activation_len = TOKENS * HIDDEN;
    let cache_len = TOKENS * KEY_VALUE_HEADS * HEAD_DIM;
    let mut expected_current = input.clone();
    let mut expected_checkpoints = Vec::new();
    let mut expected_caches = Vec::new();
    let mut calls = 0_usize;

    let trace = decoder_stack_prefill_f32(
        &input,
        config,
        &checkpoints,
        &final_norm_weight,
        |layer_index: usize,
         supplied_config: DecoderLayerConfig,
         current: &[f32]|
         -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            assert_eq!(supplied_config, config.layer);
            assert_eq!(layer_index, calls);
            assert_f32_bits(
                &format!("64-layer current input {layer_index}"),
                current,
                &expected_current,
            );
            calls += 1;
            let output = next_output(layer_index, current);
            let layer_trace = valid_minimal_trace(supplied_config, layer_index, output.clone());
            if checkpoints.contains(&layer_index) {
                expected_checkpoints.push((layer_index, output.clone()));
            }
            expected_current = output;
            expected_caches.push(layer_trace.kv_cache.clone());
            Ok(layer_trace)
        },
    )
    .unwrap();

    assert_eq!(calls, layers);
    assert_eq!(trace.executed_layers, layers);
    assert_eq!(trace.checkpoints.len(), checkpoints.len());
    assert_eq!(trace.kv_caches.len(), layers);
    assert_eq!(
        trace.retained_checkpoint_elements,
        checkpoints.len() * activation_len
    );
    assert_eq!(trace.retained_kv_elements, layers * 2 * cache_len);
    for (layer_index, values) in expected_checkpoints {
        assert_f32_bits(
            &format!("retained checkpoint {layer_index}"),
            trace.checkpoint(layer_index).unwrap(),
            &values,
        );
    }
    for layer_index in [0, 31, 63] {
        assert_cache_exact(
            &format!("retained cache {layer_index}"),
            trace.kv_cache(layer_index).unwrap(),
            &expected_caches[layer_index],
        );
    }
    assert_eq!(trace.checkpoint(1), None);
    assert_eq!(trace.checkpoint(layers), None);
    assert_eq!(trace.kv_cache(layers), None);
    let expected_final = rms_norm_f32(
        &expected_current,
        TOKENS,
        HIDDEN,
        &final_norm_weight,
        EPSILON,
    )
    .unwrap();
    assert_f32_bits("64-layer final norm", &trace.final_norm, &expected_final);
}

#[test]
fn pinned_stack_uses_exact_topology_for_one_and_four_tokens() {
    let checkpoints = [0, 9, 17];
    for tokens in [1, 4] {
        let config = pinned_layer_config(tokens);
        let input = dense(tokens * PINNED_HIDDEN, 13, 5, 67, 127.0);
        let final_norm_weight = final_norm_weight(PINNED_HIDDEN);
        let preserved = (input.clone(), final_norm_weight.clone());
        let activation_len = tokens * PINNED_HIDDEN;
        let cache_len = tokens * PINNED_KEY_VALUE_HEADS * PINNED_HEAD_DIM;
        let mut expected_current = input.clone();
        let mut expected_checkpoints = Vec::new();
        let mut expected_caches = Vec::new();
        let mut calls = 0_usize;

        let trace = pinned_decoder_stack_prefill_f32(
            &input,
            tokens,
            &checkpoints,
            &final_norm_weight,
            |layer_index: usize,
             supplied_config: DecoderLayerConfig,
             current: &[f32]|
             -> Result<DecoderLayerPrefillTrace, CpuRefError> {
                assert_eq!(supplied_config, config);
                assert_eq!(layer_index, calls);
                assert_f32_bits(
                    &format!("pinned tokens={tokens} layer={layer_index} input"),
                    current,
                    &expected_current,
                );
                calls += 1;
                let output = next_output(layer_index, current);
                let layer_trace = valid_minimal_trace(supplied_config, layer_index, output.clone());
                if checkpoints.contains(&layer_index) {
                    expected_checkpoints.push((layer_index, output.clone()));
                }
                expected_current = output;
                expected_caches.push(layer_trace.kv_cache.clone());
                Ok(layer_trace)
            },
        )
        .unwrap();

        assert_eq!(calls, PINNED_LAYERS);
        assert_eq!(trace.executed_layers, PINNED_LAYERS);
        assert_eq!(trace.checkpoints.len(), checkpoints.len());
        assert_eq!(trace.kv_caches.len(), PINNED_LAYERS);
        assert_eq!(
            trace.retained_checkpoint_elements,
            checkpoints.len() * activation_len
        );
        assert_eq!(trace.retained_kv_elements, PINNED_LAYERS * 2 * cache_len);
        for (layer_index, values) in expected_checkpoints {
            assert_f32_bits(
                &format!("pinned checkpoint {layer_index}"),
                trace.checkpoint(layer_index).unwrap(),
                &values,
            );
        }
        for (layer_index, expected_cache) in expected_caches.iter().enumerate() {
            let cache = trace.kv_cache(layer_index).unwrap();
            assert_eq!(cache.tokens, tokens);
            assert_eq!(cache.key_value_heads, PINNED_KEY_VALUE_HEADS);
            assert_eq!(cache.head_dim, PINNED_HEAD_DIM);
            assert_eq!(cache.keys.len(), cache_len);
            assert_eq!(cache.values.len(), cache_len);
            assert_cache_exact(
                &format!("pinned cache {layer_index}"),
                cache,
                expected_cache,
            );
        }
        let expected_final = rms_norm_f32(
            &expected_current,
            tokens,
            PINNED_HIDDEN,
            &final_norm_weight,
            PINNED_EPSILON,
        )
        .unwrap();
        assert_f32_bits("pinned final norm", &trace.final_norm, &expected_final);
        assert_eq!(trace.checkpoint(1), None);
        assert_eq!(trace.kv_cache(PINNED_LAYERS), None);
        assert_eq!(input, preserved.0);
        assert_eq!(final_norm_weight, preserved.1);
    }
}

#[test]
fn pinned_stack_invalid_inputs_fail_before_layer_zero() {
    let tokens = 1;
    let input = dense(PINNED_HIDDEN, 13, 5, 67, 127.0);
    let final_norm_weight = final_norm_weight(PINNED_HIDDEN);
    assert_pinned_rejected_before_layer_zero(
        "pinned zero tokens",
        &[],
        0,
        &[],
        &final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_pinned_rejected_before_layer_zero(
        "pinned overflowing tokens",
        &[],
        usize::MAX,
        &[],
        &final_norm_weight,
        CpuRefErrorCode::DimensionMismatch,
    );

    for long in [false, true] {
        let mut malformed_input = input.clone();
        if long {
            malformed_input.push(0.0);
        } else {
            malformed_input.pop();
        }
        assert_pinned_rejected_before_layer_zero(
            if long {
                "pinned long input"
            } else {
                "pinned short input"
            },
            &malformed_input,
            tokens,
            &[],
            &final_norm_weight,
            CpuRefErrorCode::DimensionMismatch,
        );
        let mut malformed_weight = final_norm_weight.clone();
        if long {
            malformed_weight.push(1.0);
        } else {
            malformed_weight.pop();
        }
        assert_pinned_rejected_before_layer_zero(
            if long {
                "pinned long final norm weight"
            } else {
                "pinned short final norm weight"
            },
            &input,
            tokens,
            &[],
            &malformed_weight,
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut malformed_input = input.clone();
        malformed_input[input.len() / 2] = value;
        assert_pinned_rejected_before_layer_zero(
            &format!("pinned non-finite input {value:?}"),
            &malformed_input,
            tokens,
            &[],
            &final_norm_weight,
            CpuRefErrorCode::NonFiniteInput,
        );
        let mut malformed_weight = final_norm_weight.clone();
        malformed_weight[final_norm_weight.len() / 2] = value;
        assert_pinned_rejected_before_layer_zero(
            &format!("pinned non-finite final norm weight {value:?}"),
            &input,
            tokens,
            &[],
            &malformed_weight,
            CpuRefErrorCode::NonFiniteInput,
        );
    }

    for (case, checkpoints) in [
        ("pinned duplicate checkpoints", &[1, 1][..]),
        ("pinned reordered checkpoints", &[2, 1][..]),
        ("pinned out-of-range checkpoint", &[0, PINNED_LAYERS][..]),
    ] {
        assert_pinned_rejected_before_layer_zero(
            case,
            &input,
            tokens,
            checkpoints,
            &final_norm_weight,
            CpuRefErrorCode::InvalidCheckpointSelection,
        );
    }
}
