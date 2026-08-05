use std::{
    collections::BTreeMap,
    env,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use pvlc_cpu_ref::decode_gqa_f32;
use pvlc_runtime_core::{
    DecoderCachedGqaStage, DecoderGqaSplitDescriptor, DecoderKvSessionDescriptor,
    DecoderKvSessionStep, KernelId,
};
use pvlc_runtime_native::{
    BackendKind, DecoderCachedGqaBindGroupEvidence, DecoderCachedGqaBindingEvidence,
    DecoderCachedGqaBufferEvidence, DecoderCachedGqaBufferRole, DecoderKvSessionEffect,
    DecoderKvSessionStepExecution, ErrorScopeKind, NativeDecoderKvSession, NativeOptions,
    NativeRuntime, RuntimeCounters, RuntimeErrorCode, RuntimeEvent, RuntimeObserver,
};
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
const KV_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIM;
const POISON_BITS: u32 = 0x7fc0_51a7;

static RUNTIME: OnceLock<Result<NativeRuntime, String>> = OnceLock::new();
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn hardware_required() -> bool {
    env_flag("PVLC_REQUIRE_NATIVE_GPU") || env_flag("PVLC_REQUIRE_M4_METAL")
}

fn runtime() -> Option<&'static NativeRuntime> {
    match RUNTIME.get_or_init(|| {
        NativeRuntime::new(NativeOptions::default()).map_err(|error| error.to_string())
    }) {
        Ok(runtime) => {
            if env_flag("PVLC_REQUIRE_M4_METAL") {
                assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
                assert!(
                    runtime.capabilities().adapter_name.contains("M4 Pro"),
                    "expected an M4 Pro adapter, got {}",
                    runtime.capabilities().adapter_name
                );
            }
            Some(runtime)
        }
        Err(error) if hardware_required() => panic!("native GPU is required: {error}"),
        Err(error) => {
            eprintln!("skipping native GPU contract because no adapter is available: {error}");
            None
        }
    }
}

fn serial_test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone)]
struct InitialCache {
    prefix_tokens: usize,
    cache_capacity: usize,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl InitialCache {
    fn new(prefix_tokens: usize, cache_capacity: usize, seed: usize) -> Self {
        assert!(prefix_tokens > 0);
        assert!(cache_capacity > prefix_tokens);
        let cache_elements = cache_capacity * KV_WIDTH;
        let prefix_elements = prefix_tokens * KV_WIDTH;
        let mut key_cache = vec![f32::from_bits(POISON_BITS); cache_elements];
        let mut value_cache = vec![f32::from_bits(POISON_BITS); cache_elements];
        for index in 0..prefix_elements {
            key_cache[index] = patterned(index, seed + 3, 0.019);
            value_cache[index] = patterned(index, seed + 11, 0.023) * 1.7;
        }
        Self {
            prefix_tokens,
            cache_capacity,
            key_cache,
            value_cache,
        }
    }

    fn descriptor(&self) -> DecoderKvSessionDescriptor<'_> {
        DecoderKvSessionDescriptor {
            query_heads: QUERY_HEADS as u32,
            key_value_heads: KEY_VALUE_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            prefix_tokens: self.prefix_tokens as u32,
            cache_capacity: self.cache_capacity as u32,
            key_cache: &self.key_cache,
            value_cache: &self.value_cache,
        }
    }
}

#[derive(Clone)]
struct OwnedStep {
    query: Vec<f32>,
    appended_key: Vec<f32>,
    appended_value: Vec<f32>,
}

impl OwnedStep {
    fn new(seed: usize) -> Self {
        Self {
            query: (0..QUERY_WIDTH)
                .map(|index| patterned(index, seed + 17, 0.013))
                .collect(),
            appended_key: (0..KV_WIDTH)
                .map(|index| patterned(index, seed + 23, 0.017))
                .collect(),
            appended_value: (0..KV_WIDTH)
                .map(|index| patterned(index, seed + 31, 0.029) * 1.3)
                .collect(),
        }
    }

    fn borrowed(&self) -> DecoderKvSessionStep<'_> {
        DecoderKvSessionStep {
            query: &self.query,
            appended_key: &self.appended_key,
            appended_value: &self.appended_value,
        }
    }
}

struct CpuSession {
    cache_tokens: usize,
    cache_capacity: usize,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl CpuSession {
    fn new(initial: &InitialCache) -> Self {
        Self {
            cache_tokens: initial.prefix_tokens,
            cache_capacity: initial.cache_capacity,
            key_cache: initial.key_cache.clone(),
            value_cache: initial.value_cache.clone(),
        }
    }

    fn step(&mut self, step: &OwnedStep) -> Vec<f32> {
        assert!(self.cache_tokens < self.cache_capacity);
        let append_start = self.cache_tokens * KV_WIDTH;
        self.key_cache[append_start..append_start + KV_WIDTH].copy_from_slice(&step.appended_key);
        self.value_cache[append_start..append_start + KV_WIDTH]
            .copy_from_slice(&step.appended_value);
        self.cache_tokens += 1;
        let valid_elements = self.cache_tokens * KV_WIDTH;
        decode_gqa_f32(
            &step.query,
            &self.key_cache[..valid_elements],
            &self.value_cache[..valid_elements],
            self.cache_tokens,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .unwrap()
    }
}

fn patterned(index: usize, seed: usize, scale: f32) -> f32 {
    (((index * 37 + seed * 19 + 7) % 4_093) as f32 * scale).sin()
}

fn policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 4.0e-4,
        max_mean_abs: 6.0e-5,
        max_p99_abs: 2.5e-4,
        max_relative_l2: 1.5e-4,
        min_cosine_similarity: 0.999_99,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    }
}

fn assert_attention_matches(expected: &[f32], actual: &[f32], label: &str) {
    let report = compare_f32(
        expected,
        actual,
        &[1, QUERY_HEADS, HEAD_DIM],
        ComparisonAxes::default(),
    )
    .unwrap();
    let verdict = report.assess(&policy()).unwrap();
    assert!(
        verdict.passed(),
        "{label} failed: {report:?}; {:?}",
        verdict.violations()
    );
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}] differs: actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn persistent_session_matches_cpu_across_steps_and_finish_preserves_the_physical_cache() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };

    for (case_index, prefix_tokens) in [1, 15, 31, 127, 332].into_iter().enumerate() {
        let initial = InitialCache::new(prefix_tokens, prefix_tokens + 5, case_index + 1);
        let original_keys = initial.key_cache.clone();
        let original_values = initial.value_cache.clone();
        let mut cpu = CpuSession::new(&initial);
        let mut session = runtime
            .begin_decoder_kv_session(&initial.descriptor())
            .unwrap();

        assert_eq!(session.cache_tokens(), prefix_tokens as u32);
        assert_eq!(session.cache_capacity(), (prefix_tokens + 5) as u32);
        for step_index in 0..3 {
            let step = OwnedStep::new(case_index * 101 + step_index * 17 + 13);
            let expected = cpu.step(&step);
            let execution = session.step(&step.borrowed()).unwrap();
            assert_eq!(execution.cache_tokens, cpu.cache_tokens as u32);
            assert_eq!(
                execution.diagnostics.cache_tokens_before,
                cpu.cache_tokens as u32 - 1
            );
            assert_eq!(
                execution.diagnostics.cache_tokens_after,
                cpu.cache_tokens as u32
            );
            assert_attention_matches(
                &expected,
                &execution.attention,
                &format!("prefix={prefix_tokens} step={step_index}"),
            );
        }

        let snapshot = session.finish().unwrap();
        assert_eq!(snapshot.cache_tokens, cpu.cache_tokens as u32);
        assert_eq!(snapshot.cache_capacity, cpu.cache_capacity as u32);
        assert_bits_eq(&snapshot.key_cache, &cpu.key_cache, "finished key cache");
        assert_bits_eq(
            &snapshot.value_cache,
            &cpu.value_cache,
            "finished value cache",
        );
        assert_bits_eq(&initial.key_cache, &original_keys, "host initial key cache");
        assert_bits_eq(
            &initial.value_cache,
            &original_values,
            "host initial value cache",
        );
    }
}

#[derive(Default)]
struct EventLog {
    events: Mutex<Vec<RuntimeEvent>>,
}

impl EventLog {
    fn snapshot(&self) -> Vec<RuntimeEvent> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl RuntimeObserver for EventLog {
    fn on_event(&self, event: RuntimeEvent) {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event);
    }
}

#[derive(Debug, Eq, PartialEq)]
enum DecoderStepRawEvent {
    QueueWrite {
        buffer_identity: u64,
        byte_offset: u64,
        byte_length: u64,
    },
    CommandEncoder {
        label: String,
    },
    ComputePass {
        pass_index: usize,
        stage: DecoderCachedGqaStage,
    },
    Dispatch {
        ordinal: usize,
        stage: DecoderCachedGqaStage,
        kernel: KernelId,
        workgroups: [u32; 3],
    },
    Copy {
        ordinal: usize,
        source_buffer_identity: u64,
        source_offset: u64,
        destination_buffer_identity: u64,
        destination_offset: u64,
        byte_length: u64,
    },
    Submit {
        command_buffers: u32,
    },
    Map {
        buffer_identity: u64,
        byte_offset: u64,
        byte_length: u64,
    },
}

fn decoder_raw_events(events: &[RuntimeEvent]) -> Vec<DecoderStepRawEvent> {
    events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::QueueBufferWritten {
                buffer_identity,
                byte_offset,
                byte_length,
                ..
            } => Some(DecoderStepRawEvent::QueueWrite {
                buffer_identity: *buffer_identity,
                byte_offset: *byte_offset,
                byte_length: *byte_length,
            }),
            RuntimeEvent::DecoderCommandEncoderCreated { label } => {
                Some(DecoderStepRawEvent::CommandEncoder {
                    label: label.clone(),
                })
            }
            RuntimeEvent::DecoderComputePassEncoded { pass_index, stage } => {
                Some(DecoderStepRawEvent::ComputePass {
                    pass_index: *pass_index,
                    stage: *stage,
                })
            }
            RuntimeEvent::DecoderDispatchEncoded {
                ordinal,
                stage,
                kernel,
                workgroups,
            } => Some(DecoderStepRawEvent::Dispatch {
                ordinal: *ordinal,
                stage: *stage,
                kernel: *kernel,
                workgroups: *workgroups,
            }),
            RuntimeEvent::DecoderBufferCopyEncoded {
                ordinal,
                source_buffer_identity,
                source_offset,
                destination_buffer_identity,
                destination_offset,
                byte_length,
            } => Some(DecoderStepRawEvent::Copy {
                ordinal: *ordinal,
                source_buffer_identity: *source_buffer_identity,
                source_offset: *source_offset,
                destination_buffer_identity: *destination_buffer_identity,
                destination_offset: *destination_offset,
                byte_length: *byte_length,
            }),
            RuntimeEvent::SubmissionQueued {
                command_buffers, ..
            } => Some(DecoderStepRawEvent::Submit {
                command_buffers: *command_buffers,
            }),
            RuntimeEvent::DecoderMapRequested {
                buffer_identity,
                byte_offset,
                byte_length,
            } => Some(DecoderStepRawEvent::Map {
                buffer_identity: *buffer_identity,
                byte_offset: *byte_offset,
                byte_length: *byte_length,
            }),
            _ => None,
        })
        .collect()
}

#[test]
fn creation_step_and_finish_have_exact_persistent_effect_topology() {
    let _guard = serial_test_guard();
    let observer = Arc::new(EventLog::default());
    let runtime = match NativeRuntime::new(NativeOptions {
        observer: Some(observer.clone()),
    }) {
        Ok(runtime) => runtime,
        Err(error) if hardware_required() => panic!("native GPU is required: {error}"),
        Err(error) => {
            eprintln!("skipping native GPU contract because no adapter is available: {error}");
            return;
        }
    };
    let initial = InitialCache::new(7, 13, 71);

    let events_before_create = observer.snapshot().len();
    let before_create = runtime.counters();
    let mut session = runtime
        .begin_decoder_kv_session(&initial.descriptor())
        .unwrap();
    let after_create = runtime.counters();
    // M7o2 amendment: the session allocates three extra split-K resources
    // (partials scratch plus two split uniforms), builds three pipelines and
    // three bind groups (append, split partial, split merge).
    assert_counter_delta(
        before_create,
        after_create,
        RuntimeCounters {
            buffer_allocations: 11,
            submissions: 0,
            pipeline_creations: 3,
            bind_group_creations: 3,
            command_encoder_creations: 0,
            dispatch_encodings: 0,
            buffer_copy_encodings: 0,
            map_requests: 0,
            queue_writes: 2,
        },
    );

    let creation = session.creation_diagnostics();
    assert_eq!(creation.initial_cache_tokens, 7);
    assert_eq!(creation.cache_capacity, 13);
    assert_eq!(
        creation.checked_error_scopes,
        [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ]
    );
    assert!(creation.captured_errors.is_empty());
    // M7o2 amendment: the session evidence covers the append kernel plus the
    // split-K GQA pair; the serial `decoder_gqa_f32` kernel is no longer part
    // of the persistent session authority.
    assert_eq!(creation.shader_blake3.len(), 3);
    for kernel in [
        KernelId::DecoderKvAppendF32,
        KernelId::DecoderGqaSplitPartialF32,
        KernelId::DecoderGqaSplitMergeF32,
    ] {
        let module = pvlc_wgsl::module(kernel).unwrap();
        assert_eq!(
            creation.shader_blake3[&kernel],
            *blake3::hash(module.source.as_bytes()).as_bytes()
        );
    }
    assert!(
        !creation
            .shader_blake3
            .contains_key(&KernelId::DecoderGqaF32),
        "the serial GQA kernel must not remain in the split-K session evidence"
    );
    assert_eq!(creation.buffers.len(), 11);
    assert_eq!(creation.bind_groups.len(), 3);
    let query_bytes = (QUERY_WIDTH * size_of::<f32>()) as u64;
    let key_value_bytes = (KV_WIDTH * size_of::<f32>()) as u64;
    let cache_bytes = (initial.cache_capacity * KV_WIDTH * size_of::<f32>()) as u64;
    let split_partials_bytes = DecoderGqaSplitDescriptor::pinned(initial.cache_capacity as u32)
        .plan()
        .unwrap()
        .partials_bytes;
    let key_cache = buffer(&creation.buffers, DecoderCachedGqaBufferRole::KeyCache);
    let value_cache = buffer(&creation.buffers, DecoderCachedGqaBufferRole::ValueCache);
    let query = buffer(&creation.buffers, DecoderCachedGqaBufferRole::Query);
    let appended_key = buffer(&creation.buffers, DecoderCachedGqaBufferRole::AppendedKey);
    let appended_value = buffer(&creation.buffers, DecoderCachedGqaBufferRole::AppendedValue);
    let append_uniform = buffer(&creation.buffers, DecoderCachedGqaBufferRole::AppendUniform);
    let split_partials = buffer(&creation.buffers, DecoderCachedGqaBufferRole::SplitPartials);
    let split_partial_uniform = buffer(
        &creation.buffers,
        DecoderCachedGqaBufferRole::SplitPartialUniform,
    );
    let split_merge_uniform = buffer(
        &creation.buffers,
        DecoderCachedGqaBufferRole::SplitMergeUniform,
    );
    let attention_output = buffer(
        &creation.buffers,
        DecoderCachedGqaBufferRole::AttentionOutput,
    );
    let readback = buffer(&creation.buffers, DecoderCachedGqaBufferRole::Readback);
    let expected_buffers = [
        (DecoderCachedGqaBufferRole::Query, query, query_bytes),
        (
            DecoderCachedGqaBufferRole::AppendedKey,
            appended_key,
            key_value_bytes,
        ),
        (
            DecoderCachedGqaBufferRole::AppendedValue,
            appended_value,
            key_value_bytes,
        ),
        (DecoderCachedGqaBufferRole::KeyCache, key_cache, cache_bytes),
        (
            DecoderCachedGqaBufferRole::ValueCache,
            value_cache,
            cache_bytes,
        ),
        (
            DecoderCachedGqaBufferRole::AttentionOutput,
            attention_output,
            query_bytes,
        ),
        (
            DecoderCachedGqaBufferRole::AppendUniform,
            append_uniform,
            16,
        ),
        (
            DecoderCachedGqaBufferRole::SplitPartials,
            split_partials,
            split_partials_bytes,
        ),
        (
            DecoderCachedGqaBufferRole::SplitPartialUniform,
            split_partial_uniform,
            16,
        ),
        (
            DecoderCachedGqaBufferRole::SplitMergeUniform,
            split_merge_uniform,
            16,
        ),
        (DecoderCachedGqaBufferRole::Readback, readback, query_bytes),
    ];
    assert_eq!(
        creation
            .buffers
            .iter()
            .map(|buffer| { (buffer.role, buffer.buffer_identity, buffer.allocation_bytes,) })
            .collect::<Vec<_>>(),
        expected_buffers
    );
    for (index, (_, identity, _)) in expected_buffers.iter().enumerate() {
        for (_, other_identity, _) in &expected_buffers[index + 1..] {
            assert_ne!(
                identity, other_identity,
                "all eleven persistent allocations need distinct device identities"
            );
        }
    }
    assert_eq!(
        creation.bind_groups,
        vec![
            DecoderCachedGqaBindGroupEvidence {
                stage: DecoderCachedGqaStage::AppendKeyValue,
                bindings: vec![
                    DecoderCachedGqaBindingEvidence {
                        binding: 0,
                        buffer_identity: appended_key,
                        byte_offset: 0,
                        byte_length: key_value_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 1,
                        buffer_identity: appended_value,
                        byte_offset: 0,
                        byte_length: key_value_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 2,
                        buffer_identity: key_cache,
                        byte_offset: 0,
                        byte_length: cache_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 3,
                        buffer_identity: value_cache,
                        byte_offset: 0,
                        byte_length: cache_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 4,
                        buffer_identity: append_uniform,
                        byte_offset: 0,
                        byte_length: 16,
                    },
                ],
            },
            DecoderCachedGqaBindGroupEvidence {
                stage: DecoderCachedGqaStage::SplitGqaPartial,
                bindings: vec![
                    DecoderCachedGqaBindingEvidence {
                        binding: 0,
                        buffer_identity: query,
                        byte_offset: 0,
                        byte_length: query_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 1,
                        buffer_identity: key_cache,
                        byte_offset: 0,
                        byte_length: cache_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 2,
                        buffer_identity: value_cache,
                        byte_offset: 0,
                        byte_length: cache_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 3,
                        buffer_identity: split_partials,
                        byte_offset: 0,
                        byte_length: split_partials_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 4,
                        buffer_identity: split_partial_uniform,
                        byte_offset: 0,
                        byte_length: 16,
                    },
                ],
            },
            // M7o2 amendment: the merge shader declares the cache bindings
            // but never reads them, so the derived native layout only covers
            // the statically used partials, output, and uniform entries.
            DecoderCachedGqaBindGroupEvidence {
                stage: DecoderCachedGqaStage::SplitGqaMerge,
                bindings: vec![
                    DecoderCachedGqaBindingEvidence {
                        binding: 0,
                        buffer_identity: split_partials,
                        byte_offset: 0,
                        byte_length: split_partials_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 3,
                        buffer_identity: attention_output,
                        byte_offset: 0,
                        byte_length: query_bytes,
                    },
                    DecoderCachedGqaBindingEvidence {
                        binding: 4,
                        buffer_identity: split_merge_uniform,
                        byte_offset: 0,
                        byte_length: 16,
                    },
                ],
            },
        ]
    );
    let creation_writes = observer.snapshot()[events_before_create..]
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::QueueBufferWritten {
                buffer_identity,
                byte_offset,
                byte_length,
                ..
            } => Some((*buffer_identity, *byte_offset, *byte_length)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        creation_writes,
        vec![(key_cache, 0, cache_bytes), (value_cache, 0, cache_bytes)],
        "each compact initial cache must be uploaded exactly once at creation"
    );

    // M7o2 amendment: the serial attention uniform write is replaced by the
    // two split-K uniform writes.
    let expected_writes = [
        (DecoderCachedGqaBufferRole::Query, query, query_bytes),
        (
            DecoderCachedGqaBufferRole::AppendedKey,
            appended_key,
            key_value_bytes,
        ),
        (
            DecoderCachedGqaBufferRole::AppendedValue,
            appended_value,
            key_value_bytes,
        ),
        (
            DecoderCachedGqaBufferRole::AppendUniform,
            append_uniform,
            16,
        ),
        (
            DecoderCachedGqaBufferRole::SplitPartialUniform,
            split_partial_uniform,
            16,
        ),
        (
            DecoderCachedGqaBufferRole::SplitMergeUniform,
            split_merge_uniform,
            16,
        ),
    ];

    let step = OwnedStep::new(811);
    let execution = execute_and_assert_step_effect_topology(
        &runtime,
        observer.as_ref(),
        &mut session,
        &step.borrowed(),
        7,
        &expected_writes,
        key_cache,
        value_cache,
        attention_output,
        readback,
    );

    let second = OwnedStep::new(977);
    let second_execution = execute_and_assert_step_effect_topology(
        &runtime,
        observer.as_ref(),
        &mut session,
        &second.borrowed(),
        8,
        &expected_writes,
        key_cache,
        value_cache,
        attention_output,
        readback,
    );
    assert_eq!(
        second_execution.diagnostics.effects, execution.diagnostics.effects,
        "every step must reuse the complete persistent effect topology"
    );

    let events_before_finish = observer.snapshot().len();
    let before_finish = runtime.counters();
    let snapshot = session.finish().unwrap();
    let after_finish = runtime.counters();
    assert_counter_delta(
        before_finish,
        after_finish,
        RuntimeCounters {
            buffer_allocations: 1,
            submissions: 1,
            pipeline_creations: 0,
            bind_group_creations: 0,
            command_encoder_creations: 1,
            dispatch_encodings: 0,
            buffer_copy_encodings: 2,
            map_requests: 1,
            queue_writes: 0,
        },
    );
    assert_eq!(
        snapshot.diagnostics.readback_bytes,
        (2 * initial.cache_capacity * KV_WIDTH * size_of::<f32>()) as u64
    );
    assert_eq!(snapshot.diagnostics.copy_count, 2);
    assert_eq!(snapshot.diagnostics.command_buffer_count, 1);
    assert_eq!(snapshot.diagnostics.submission_count, 1);
    assert_eq!(snapshot.diagnostics.map_count, 1);
    let finish_readback = snapshot.diagnostics.readback_buffer_identity;
    for (_, persistent_identity, _) in expected_buffers {
        assert_ne!(finish_readback, persistent_identity);
    }
    let finish_events = observer.snapshot();
    let actual_finish_raw = decoder_raw_events(&finish_events[events_before_finish..]);
    assert_eq!(
        actual_finish_raw,
        vec![
            DecoderStepRawEvent::CommandEncoder {
                label: "decoder-kv-session-finish-encoder".to_owned(),
            },
            DecoderStepRawEvent::Copy {
                ordinal: 1,
                source_buffer_identity: key_cache,
                source_offset: 0,
                destination_buffer_identity: finish_readback,
                destination_offset: 0,
                byte_length: cache_bytes,
            },
            DecoderStepRawEvent::Copy {
                ordinal: 2,
                source_buffer_identity: value_cache,
                source_offset: 0,
                destination_buffer_identity: finish_readback,
                destination_offset: cache_bytes,
                byte_length: cache_bytes,
            },
            DecoderStepRawEvent::Submit { command_buffers: 1 },
            DecoderStepRawEvent::Map {
                buffer_identity: finish_readback,
                byte_offset: 0,
                byte_length: cache_bytes * 2,
            },
        ]
    );
}

struct PanicAfterSubmission {
    armed: AtomicBool,
}

impl PanicAfterSubmission {
    fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
        }
    }

    fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl RuntimeObserver for PanicAfterSubmission {
    fn on_event(&self, event: RuntimeEvent) {
        if matches!(event, RuntimeEvent::SubmissionQueued { .. })
            && self.armed.swap(false, Ordering::SeqCst)
        {
            panic!("injected observer failure after queue submission");
        }
    }
}

#[test]
fn unwinding_after_submission_terminally_poisons_the_session_before_logical_commit() {
    let _guard = serial_test_guard();
    let observer = Arc::new(PanicAfterSubmission::new());
    let runtime = match NativeRuntime::new(NativeOptions {
        observer: Some(observer.clone()),
    }) {
        Ok(runtime) => runtime,
        Err(error) if hardware_required() => panic!("native GPU is required: {error}"),
        Err(error) => {
            eprintln!("skipping native GPU contract because no adapter is available: {error}");
            return;
        }
    };
    let initial = InitialCache::new(3, 7, 1_701);
    let mut session = runtime
        .begin_decoder_kv_session(&initial.descriptor())
        .unwrap();
    let step = OwnedStep::new(1_803);

    observer.arm();
    let before_failure = runtime.counters();
    let panic_result = catch_unwind(AssertUnwindSafe(|| session.step(&step.borrowed())));
    assert!(
        panic_result.is_err(),
        "the observer must interrupt after a real queue submission"
    );
    // M7o2 amendment: the interrupted step encodes three dispatches and
    // writes six operands before the observer unwinds.
    let after_failure = runtime.counters();
    assert_counter_delta(
        before_failure,
        after_failure,
        RuntimeCounters {
            buffer_allocations: 0,
            submissions: 1,
            pipeline_creations: 0,
            bind_group_creations: 0,
            command_encoder_creations: 1,
            dispatch_encodings: 3,
            buffer_copy_encodings: 1,
            map_requests: 0,
            queue_writes: 6,
        },
    );
    assert_eq!(
        session.cache_tokens(),
        initial.prefix_tokens as u32,
        "logical cache length must not commit across post-submit unwinding"
    );

    let before_rejected_step = runtime.counters();
    let step_error = session.step(&OwnedStep::new(1_907).borrowed()).unwrap_err();
    assert_eq!(step_error.code(), RuntimeErrorCode::Operation);
    assert!(step_error.to_string().contains("poisoned"));
    assert_eq!(runtime.counters(), before_rejected_step);

    let before_rejected_finish = runtime.counters();
    let finish_error = session.finish().unwrap_err();
    assert_eq!(finish_error.code(), RuntimeErrorCode::Operation);
    assert!(finish_error.to_string().contains("poisoned"));
    assert_eq!(runtime.counters(), before_rejected_finish);
}

#[test]
fn invalid_steps_have_zero_effect_and_recover_before_capacity_becomes_terminal() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };
    let initial = InitialCache::new(3, 5, 303);
    let mut cpu = CpuSession::new(&initial);
    let mut session = runtime
        .begin_decoder_kv_session(&initial.descriptor())
        .unwrap();
    let valid = OwnedStep::new(404);

    for operand in 0..3 {
        let original = match operand {
            0 => &valid.query,
            1 => &valid.appended_key,
            2 => &valid.appended_value,
            _ => unreachable!(),
        };
        for values in wrong_lengths(original) {
            let step = match operand {
                0 => DecoderKvSessionStep {
                    query: &values,
                    ..valid.borrowed()
                },
                1 => DecoderKvSessionStep {
                    appended_key: &values,
                    ..valid.borrowed()
                },
                2 => DecoderKvSessionStep {
                    appended_value: &values,
                    ..valid.borrowed()
                },
                _ => unreachable!(),
            };
            assert_step_preflight_no_effect(runtime, &mut session, &step, 3);
        }
    }
    for operand in 0..3 {
        let mut poisoned = valid.clone();
        match operand {
            0 => poisoned.query[7] = f32::NAN,
            1 => poisoned.appended_key[11] = f32::INFINITY,
            2 => poisoned.appended_value[13] = f32::NEG_INFINITY,
            _ => unreachable!(),
        }
        assert_step_preflight_no_effect(runtime, &mut session, &poisoned.borrowed(), 3);
    }

    let first = OwnedStep::new(505);
    let expected = cpu.step(&first);
    let execution = session.step(&first.borrowed()).unwrap();
    assert_attention_matches(
        &expected,
        &execution.attention,
        "first recovered valid step",
    );
    assert_eq!(session.cache_tokens(), 4);

    let mut invalid_after_append = OwnedStep::new(559);
    invalid_after_append.appended_value[17] = f32::NAN;
    assert_step_preflight_no_effect(runtime, &mut session, &invalid_after_append.borrowed(), 4);

    let second = OwnedStep::new(606);
    let expected = cpu.step(&second);
    let execution = session.step(&second.borrowed()).unwrap();
    assert_attention_matches(
        &expected,
        &execution.attention,
        "valid continuation after post-append rejection",
    );
    assert_eq!(session.cache_tokens(), 5);
    let before_exhausted = runtime.counters();
    let error = session.step(&OwnedStep::new(707).borrowed()).unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
    assert_eq!(runtime.counters(), before_exhausted);
    assert_eq!(session.cache_tokens(), 5);

    let snapshot = session.finish().unwrap();
    assert_bits_eq(&snapshot.key_cache, &cpu.key_cache, "recovered key cache");
    assert_bits_eq(
        &snapshot.value_cache,
        &cpu.value_cache,
        "recovered value cache",
    );
}

#[test]
fn interleaved_sessions_keep_distinct_device_cache_authority() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };
    let initial_a = InitialCache::new(2, 7, 901);
    let initial_b = InitialCache::new(4, 9, 1901);
    let mut cpu_a = CpuSession::new(&initial_a);
    let mut cpu_b = CpuSession::new(&initial_b);
    let mut session_a = runtime
        .begin_decoder_kv_session(&initial_a.descriptor())
        .unwrap();
    let mut session_b = runtime
        .begin_decoder_kv_session(&initial_b.descriptor())
        .unwrap();
    let a_key_identity = buffer(
        &session_a.creation_diagnostics().buffers,
        DecoderCachedGqaBufferRole::KeyCache,
    );
    let b_key_identity = buffer(
        &session_b.creation_diagnostics().buffers,
        DecoderCachedGqaBufferRole::KeyCache,
    );
    assert_ne!(a_key_identity, b_key_identity);

    for (session_id, seed) in [(0, 17), (1, 29), (0, 43), (1, 59), (0, 71)] {
        let step = OwnedStep::new(seed);
        let (expected, execution) = if session_id == 0 {
            (cpu_a.step(&step), session_a.step(&step.borrowed()).unwrap())
        } else {
            (cpu_b.step(&step), session_b.step(&step.borrowed()).unwrap())
        };
        assert_attention_matches(&expected, &execution.attention, "interleaved session");
    }

    let snapshot_a = session_a.finish().unwrap();
    let snapshot_b = session_b.finish().unwrap();
    assert_bits_eq(&snapshot_a.key_cache, &cpu_a.key_cache, "session A key");
    assert_bits_eq(
        &snapshot_a.value_cache,
        &cpu_a.value_cache,
        "session A value",
    );
    assert_bits_eq(&snapshot_b.key_cache, &cpu_b.key_cache, "session B key");
    assert_bits_eq(
        &snapshot_b.value_cache,
        &cpu_b.value_cache,
        "session B value",
    );
}

#[test]
fn shader_overrides_prove_persistent_append_and_gqa_are_both_causal() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };
    let initial = InitialCache::new(3, 8, 73);
    let append_module = pvlc_wgsl::module(KernelId::DecoderKvAppendF32).unwrap();
    let append_source = append_module
        .source
        .replace(
            "key_cache.data[cache_index] = appended_key.data[linear];",
            "key_cache.data[cache_index] = appended_key.data[linear] + 4.0;",
        )
        .replace(
            "value_cache.data[cache_index] = appended_value.data[linear];",
            "value_cache.data[cache_index] = appended_value.data[linear] - 3.0;",
        );
    assert_ne!(append_source, append_module.source);
    validate_override(append_module, &append_source);
    let append_overrides = BTreeMap::from([(KernelId::DecoderKvAppendF32, append_source.clone())]);
    let mut append_session = runtime
        .begin_decoder_kv_session_with_shader_overrides(&initial.descriptor(), &append_overrides)
        .unwrap();
    let mut expected_keys = initial.key_cache.clone();
    let mut expected_values = initial.value_cache.clone();
    let mut cache_tokens = initial.prefix_tokens;
    for seed in [311, 419] {
        let step = OwnedStep::new(seed);
        let append_start = cache_tokens * KV_WIDTH;
        for index in 0..KV_WIDTH {
            expected_keys[append_start + index] = step.appended_key[index] + 4.0;
            expected_values[append_start + index] = step.appended_value[index] - 3.0;
        }
        cache_tokens += 1;
        let valid_elements = cache_tokens * KV_WIDTH;
        let oracle = decode_gqa_f32(
            &step.query,
            &expected_keys[..valid_elements],
            &expected_values[..valid_elements],
            cache_tokens,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .unwrap();
        let execution = append_session.step(&step.borrowed()).unwrap();
        assert_attention_matches(
            &oracle,
            &execution.attention,
            "mutated append must feed persistent GQA",
        );
    }
    let append_snapshot = append_session.finish().unwrap();
    assert_bits_eq(
        &append_snapshot.key_cache,
        &expected_keys,
        "mutated persistent keys",
    );
    assert_bits_eq(
        &append_snapshot.value_cache,
        &expected_values,
        "mutated persistent values",
    );

    // M7o2 amendment: the persistent session executes the split-K merge
    // kernel instead of the serial GQA kernel, so causality is proven by
    // overriding the merge output; overriding the serial kernel is now
    // rejected at session creation.
    let merge_module = pvlc_wgsl::module(KernelId::DecoderGqaSplitMergeF32).unwrap();
    let merge_source = merge_module.source.replace(
        "output.data[linear] = weighted / denominator;",
        "output.data[linear] = 8192.0 + f32(linear);",
    );
    assert_ne!(merge_source, merge_module.source);
    validate_override(merge_module, &merge_source);
    let merge_overrides =
        BTreeMap::from([(KernelId::DecoderGqaSplitMergeF32, merge_source.clone())]);
    let mut merge_session = runtime
        .begin_decoder_kv_session_with_shader_overrides(&initial.descriptor(), &merge_overrides)
        .unwrap();
    for seed in [521, 607] {
        let execution = merge_session
            .step(&OwnedStep::new(seed).borrowed())
            .unwrap();
        for (index, actual) in execution.attention.iter().enumerate() {
            assert_eq!(actual.to_bits(), (8192.0 + index as f32).to_bits());
        }
    }
    assert_eq!(
        merge_session.creation_diagnostics().shader_blake3[&KernelId::DecoderKvAppendF32],
        *blake3::hash(append_module.source.as_bytes()).as_bytes()
    );
    assert_eq!(
        merge_session.creation_diagnostics().shader_blake3[&KernelId::DecoderGqaSplitMergeF32],
        *blake3::hash(merge_source.as_bytes()).as_bytes()
    );

    let serial_module = pvlc_wgsl::module(KernelId::DecoderGqaF32).unwrap();
    let serial_overrides =
        BTreeMap::from([(KernelId::DecoderGqaF32, serial_module.source.to_owned())]);
    let before_rejected = runtime.counters();
    let rejected = runtime
        .begin_decoder_kv_session_with_shader_overrides(&initial.descriptor(), &serial_overrides)
        .unwrap_err();
    assert_eq!(rejected.code(), RuntimeErrorCode::Validation);
    assert!(
        rejected
            .to_string()
            .contains("is not used by decoder KV session"),
        "unexpected rejection: {rejected}"
    );
    assert_eq!(runtime.counters(), before_rejected);
}

#[test]
fn invalid_session_creation_fails_before_every_gpu_side_effect() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };
    let initial = InitialCache::new(3, 7, 37);
    let base = initial.descriptor();

    for descriptor in [
        DecoderKvSessionDescriptor {
            query_heads: 0,
            ..base
        },
        DecoderKvSessionDescriptor {
            prefix_tokens: base.cache_capacity,
            ..base
        },
        DecoderKvSessionDescriptor {
            cache_capacity: u32::MAX,
            key_cache: &[],
            value_cache: &[],
            ..base
        },
    ] {
        assert_creation_preflight_no_effect(runtime, descriptor);
    }
    for (query_heads, key_value_heads, head_dim) in [
        (8, KEY_VALUE_HEADS as u32, HEAD_DIM as u32),
        (QUERY_HEADS as u32, 1, HEAD_DIM as u32),
        (QUERY_HEADS as u32, KEY_VALUE_HEADS as u32, 64),
        (
            QUERY_HEADS as u32,
            KEY_VALUE_HEADS as u32,
            HEAD_DIM as u32 + 1,
        ),
    ] {
        let cache_elements =
            base.cache_capacity as usize * key_value_heads as usize * head_dim as usize;
        let keys = vec![0.25; cache_elements];
        let values = vec![-0.5; cache_elements];
        assert_creation_preflight_no_effect(
            runtime,
            DecoderKvSessionDescriptor {
                query_heads,
                key_value_heads,
                head_dim,
                key_cache: &keys,
                value_cache: &values,
                ..base
            },
        );
    }
    for cache_operand in 0..2 {
        let original = if cache_operand == 0 {
            &initial.key_cache
        } else {
            &initial.value_cache
        };
        for values in wrong_lengths(original) {
            let descriptor = if cache_operand == 0 {
                DecoderKvSessionDescriptor {
                    key_cache: &values,
                    ..base
                }
            } else {
                DecoderKvSessionDescriptor {
                    value_cache: &values,
                    ..base
                }
            };
            assert_creation_preflight_no_effect(runtime, descriptor);
        }
    }
    for key_operand in [true, false] {
        let mut keys = initial.key_cache.clone();
        let mut values = initial.value_cache.clone();
        if key_operand {
            keys[initial.prefix_tokens * KV_WIDTH - 1] = f32::NAN;
        } else {
            values[initial.prefix_tokens * KV_WIDTH - 1] = f32::INFINITY;
        }
        assert_creation_preflight_no_effect(
            runtime,
            DecoderKvSessionDescriptor {
                key_cache: &keys,
                value_cache: &values,
                ..base
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_and_assert_step_effect_topology(
    runtime: &NativeRuntime,
    observer: &EventLog,
    session: &mut NativeDecoderKvSession<'_>,
    step: &DecoderKvSessionStep<'_>,
    cache_tokens_before: u32,
    expected_writes: &[(DecoderCachedGqaBufferRole, u64, u64); 6],
    key_cache_identity: u64,
    value_cache_identity: u64,
    attention_output_identity: u64,
    readback_identity: u64,
) -> DecoderKvSessionStepExecution {
    let events_before_step = observer.snapshot().len();
    let before_step = runtime.counters();
    let execution = session.step(step).unwrap();
    let after_step = runtime.counters();
    // M7o2 amendment: one step now encodes three dispatches (append, split
    // partial, split merge) and writes six operands.
    assert_counter_delta(
        before_step,
        after_step,
        RuntimeCounters {
            buffer_allocations: 0,
            submissions: 1,
            pipeline_creations: 0,
            bind_group_creations: 0,
            command_encoder_creations: 1,
            dispatch_encodings: 3,
            buffer_copy_encodings: 1,
            map_requests: 1,
            queue_writes: 6,
        },
    );
    assert_eq!(
        execution.diagnostics.cache_tokens_before,
        cache_tokens_before
    );
    assert_eq!(
        execution.diagnostics.cache_tokens_after,
        cache_tokens_before + 1
    );
    assert_eq!(execution.cache_tokens, cache_tokens_before + 1);
    assert_eq!(
        execution.diagnostics.readback_bytes,
        (QUERY_WIDTH * size_of::<f32>()) as u64
    );
    assert_eq!(execution.diagnostics.dispatch_count, 3);
    assert_eq!(execution.diagnostics.compute_pass_count, 3);
    assert_eq!(execution.diagnostics.command_buffer_count, 1);
    assert_eq!(execution.diagnostics.copy_count, 1);
    assert_eq!(execution.diagnostics.submission_count, 1);
    assert_eq!(execution.diagnostics.map_count, 1);
    assert!(execution.diagnostics.queue_wall_time_ns > 0);
    assert_eq!(execution.diagnostics.effects.len(), 12);
    for (index, (role, identity, bytes)) in expected_writes.iter().copied().enumerate() {
        assert_eq!(
            execution.diagnostics.effects[index],
            DecoderKvSessionEffect::QueueWrite {
                ordinal: index + 1,
                role,
                buffer_identity: identity,
                byte_offset: 0,
                byte_length: bytes,
            }
        );
        assert_ne!(identity, key_cache_identity);
        assert_ne!(identity, value_cache_identity);
    }
    let split_chunk_count = (cache_tokens_before + 1).div_ceil(32);
    let split_partial_workgroups = [16 * split_chunk_count, 1, 1];
    assert_eq!(
        execution.diagnostics.effects[6],
        DecoderKvSessionEffect::Dispatch {
            ordinal: 7,
            stage: DecoderCachedGqaStage::AppendKeyValue,
            kernel: KernelId::DecoderKvAppendF32,
            workgroups: [4, 1, 1],
        }
    );
    assert_eq!(
        execution.diagnostics.effects[7],
        DecoderKvSessionEffect::Dispatch {
            ordinal: 8,
            stage: DecoderCachedGqaStage::SplitGqaPartial,
            kernel: KernelId::DecoderGqaSplitPartialF32,
            workgroups: split_partial_workgroups,
        }
    );
    assert_eq!(
        execution.diagnostics.effects[8],
        DecoderKvSessionEffect::Dispatch {
            ordinal: 9,
            stage: DecoderCachedGqaStage::SplitGqaMerge,
            kernel: KernelId::DecoderGqaSplitMergeF32,
            workgroups: [32, 1, 1],
        }
    );
    assert_eq!(
        execution.diagnostics.effects[9],
        DecoderKvSessionEffect::CopyAttention {
            ordinal: 10,
            source_buffer_identity: attention_output_identity,
            destination_buffer_identity: readback_identity,
            byte_length: (QUERY_WIDTH * size_of::<f32>()) as u64,
        }
    );
    assert_eq!(
        execution.diagnostics.effects[10],
        DecoderKvSessionEffect::Submit {
            ordinal: 11,
            command_buffer_count: 1,
        }
    );
    assert_eq!(
        execution.diagnostics.effects[11],
        DecoderKvSessionEffect::MapAttention {
            ordinal: 12,
            buffer_identity: readback_identity,
            byte_length: (QUERY_WIDTH * size_of::<f32>()) as u64,
        }
    );

    let raw_events = observer.snapshot();
    let raw_events = &raw_events[events_before_step..];
    let actual_raw = decoder_raw_events(raw_events);
    let mut expected_raw = Vec::with_capacity(15);
    for (_, identity, bytes) in expected_writes {
        expected_raw.push(DecoderStepRawEvent::QueueWrite {
            buffer_identity: *identity,
            byte_offset: 0,
            byte_length: *bytes,
        });
    }
    expected_raw.push(DecoderStepRawEvent::CommandEncoder {
        label: "decoder-kv-session-step-encoder".to_owned(),
    });
    expected_raw.push(DecoderStepRawEvent::ComputePass {
        pass_index: 1,
        stage: DecoderCachedGqaStage::AppendKeyValue,
    });
    expected_raw.push(DecoderStepRawEvent::Dispatch {
        ordinal: 7,
        stage: DecoderCachedGqaStage::AppendKeyValue,
        kernel: KernelId::DecoderKvAppendF32,
        workgroups: [4, 1, 1],
    });
    expected_raw.push(DecoderStepRawEvent::ComputePass {
        pass_index: 2,
        stage: DecoderCachedGqaStage::SplitGqaPartial,
    });
    expected_raw.push(DecoderStepRawEvent::Dispatch {
        ordinal: 8,
        stage: DecoderCachedGqaStage::SplitGqaPartial,
        kernel: KernelId::DecoderGqaSplitPartialF32,
        workgroups: split_partial_workgroups,
    });
    expected_raw.push(DecoderStepRawEvent::ComputePass {
        pass_index: 3,
        stage: DecoderCachedGqaStage::SplitGqaMerge,
    });
    expected_raw.push(DecoderStepRawEvent::Dispatch {
        ordinal: 9,
        stage: DecoderCachedGqaStage::SplitGqaMerge,
        kernel: KernelId::DecoderGqaSplitMergeF32,
        workgroups: [32, 1, 1],
    });
    expected_raw.push(DecoderStepRawEvent::Copy {
        ordinal: 10,
        source_buffer_identity: attention_output_identity,
        source_offset: 0,
        destination_buffer_identity: readback_identity,
        destination_offset: 0,
        byte_length: (QUERY_WIDTH * size_of::<f32>()) as u64,
    });
    expected_raw.push(DecoderStepRawEvent::Submit { command_buffers: 1 });
    expected_raw.push(DecoderStepRawEvent::Map {
        buffer_identity: readback_identity,
        byte_offset: 0,
        byte_length: (QUERY_WIDTH * size_of::<f32>()) as u64,
    });
    assert_eq!(actual_raw, expected_raw);

    execution
}

fn assert_counter_delta(
    before: RuntimeCounters,
    after: RuntimeCounters,
    expected: RuntimeCounters,
) {
    assert_eq!(
        RuntimeCounters {
            buffer_allocations: after.buffer_allocations - before.buffer_allocations,
            submissions: after.submissions - before.submissions,
            pipeline_creations: after.pipeline_creations - before.pipeline_creations,
            bind_group_creations: after.bind_group_creations - before.bind_group_creations,
            command_encoder_creations: after.command_encoder_creations
                - before.command_encoder_creations,
            dispatch_encodings: after.dispatch_encodings - before.dispatch_encodings,
            buffer_copy_encodings: after.buffer_copy_encodings - before.buffer_copy_encodings,
            map_requests: after.map_requests - before.map_requests,
            queue_writes: after.queue_writes - before.queue_writes,
        },
        expected
    );
}

fn buffer(buffers: &[DecoderCachedGqaBufferEvidence], role: DecoderCachedGqaBufferRole) -> u64 {
    buffers
        .iter()
        .find(|buffer| buffer.role == role)
        .unwrap()
        .buffer_identity
}

fn assert_step_preflight_no_effect(
    runtime: &NativeRuntime,
    session: &mut pvlc_runtime_native::NativeDecoderKvSession<'_>,
    step: &DecoderKvSessionStep<'_>,
    expected_cache_tokens: u32,
) {
    let before = runtime.counters();
    let error = session.step(step).unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
    assert_eq!(runtime.counters(), before, "{error}");
    assert_eq!(session.cache_tokens(), expected_cache_tokens);
}

fn assert_creation_preflight_no_effect(
    runtime: &NativeRuntime,
    descriptor: DecoderKvSessionDescriptor<'_>,
) {
    let before = runtime.counters();
    let error = runtime.begin_decoder_kv_session(&descriptor).unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
    assert_eq!(runtime.counters(), before, "{error}");
}

fn wrong_lengths(values: &[f32]) -> [Vec<f32>; 2] {
    let mut short = values.to_vec();
    short.pop();
    let mut long = values.to_vec();
    long.push(0.0);
    [short, long]
}

fn validate_override(module: &pvlc_wgsl::KernelModule, source: &str) {
    pvlc_wgsl::validate_source_contract(&module.spec, source).unwrap();
}
