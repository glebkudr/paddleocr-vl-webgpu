use std::{collections::BTreeMap, fmt, sync::atomic::Ordering, time::Instant};

use pvlc_runtime_core::{
    ComputeDispatchLimits, DecoderCachedGqaStage, DecoderGqaSplitDescriptor,
    DecoderKvSessionDescriptor, DecoderKvSessionPlan, DecoderKvSessionStep,
    DecoderKvSessionStepPlan, InvocationPlan, KernelId,
};

use super::{
    CHECKED_SCOPE_ORDER, DecoderCachedGqaBindGroupEvidence, DecoderCachedGqaBufferRole,
    DecoderKvSessionCreationDiagnostics, DecoderKvSessionEffect, DecoderKvSessionSnapshot,
    DecoderKvSessionSnapshotDiagnostics, DecoderKvSessionStepDiagnostics,
    DecoderKvSessionStepExecution, GPU_WAIT_TIMEOUT, NativeRuntime, RuntimeError, RuntimeErrorCode,
    RuntimeEvent, WgpuScopeDriver, await_mapping, buffer_identity, decoder_binding_evidence,
    decoder_buffer_evidence, drive_error_scopes, elapsed_ns, map_read, read_f32_buffer,
};

pub(crate) enum DecoderKvSessionState {
    Healthy,
    Poisoned,
}

pub struct NativeDecoderKvSession<'runtime> {
    runtime: &'runtime NativeRuntime,
    plan: DecoderKvSessionPlan,
    state: DecoderKvSessionState,
    cache_tokens: u32,
    query_buffer: Box<wgpu::Buffer>,
    appended_key_buffer: Box<wgpu::Buffer>,
    appended_value_buffer: Box<wgpu::Buffer>,
    key_cache_buffer: Box<wgpu::Buffer>,
    value_cache_buffer: Box<wgpu::Buffer>,
    attention_output_buffer: Box<wgpu::Buffer>,
    append_uniform_buffer: Box<wgpu::Buffer>,
    // The split-K partials scratch plane is consumed only through the two
    // split bind groups, yet the session must retain it for its whole
    // lifetime so the recorded evidence identities stay unique and the
    // device allocation stays owned by the session.
    _split_partials_buffer: Box<wgpu::Buffer>,
    split_partial_uniform_buffer: Box<wgpu::Buffer>,
    split_merge_uniform_buffer: Box<wgpu::Buffer>,
    attention_readback_buffer: Box<wgpu::Buffer>,
    append_pipeline: wgpu::ComputePipeline,
    split_partial_pipeline: wgpu::ComputePipeline,
    split_merge_pipeline: wgpu::ComputePipeline,
    append_bind_group: wgpu::BindGroup,
    split_partial_bind_group: wgpu::BindGroup,
    split_merge_bind_group: wgpu::BindGroup,
    creation_diagnostics: DecoderKvSessionCreationDiagnostics,
}

impl fmt::Debug for NativeDecoderKvSession<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeDecoderKvSession")
            .field("cache_tokens", &self.cache_tokens)
            .field("cache_capacity", &self.plan.cache_capacity)
            .finish_non_exhaustive()
    }
}

pub(super) fn begin<'runtime>(
    runtime: &'runtime NativeRuntime,
    descriptor: &DecoderKvSessionDescriptor<'_>,
    shader_overrides: &BTreeMap<KernelId, String>,
) -> Result<NativeDecoderKvSession<'runtime>, RuntimeError> {
    let plan = descriptor.plan().map_err(|error| {
        RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
    })?;
    let sources = runtime.validated_decoder_kv_session_sources(shader_overrides)?;
    validate_capabilities(runtime, &plan)?;

    let _execution = runtime
        .execution_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut scopes = WgpuScopeDriver::new(runtime.device.clone(), runtime.observer.clone());
    drive_error_scopes(&mut scopes, || {
        create_session(runtime, descriptor, plan, &sources, shader_overrides)
    })
    .map_err(|error| error.with_context("decoder KV session creation"))
}

fn validate_capabilities(
    runtime: &NativeRuntime,
    plan: &DecoderKvSessionPlan,
) -> Result<(), RuntimeError> {
    if runtime.capabilities.max_storage_buffers_per_shader_stage < 4 {
        return Err(RuntimeError::new(
            RuntimeErrorCode::Validation,
            None,
            "decoder KV session requires four storage buffers per shader stage",
        ));
    }
    let limits = ComputeDispatchLimits {
        max_workgroup_size: [
            runtime.capabilities.max_compute_workgroup_size_x,
            runtime.capabilities.max_compute_workgroup_size_y,
            runtime.capabilities.max_compute_workgroup_size_z,
        ],
        max_invocations_per_workgroup: runtime.capabilities.max_compute_invocations_per_workgroup,
        max_workgroups_per_dimension: runtime.capabilities.max_compute_workgroups_per_dimension,
    };
    // M7o2 amendment: the widest split-K dispatches occur at the full cache
    // capacity, so capability validation plans the split pair there.
    let split_capacity = DecoderGqaSplitDescriptor::pinned(plan.cache_capacity)
        .plan()
        .map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
    for invocation in [
        plan.append_invocation,
        split_capacity.partial_invocation,
        split_capacity.merge_invocation,
    ] {
        limits.validate(&invocation).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
        })?;
    }
    let key_value_row_bytes = u64::try_from(plan.key_value_width)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidInvocation,
                None,
                "decoder KV row byte size overflowed",
            )
        })?;
    for (label, bytes) in [
        ("decoder session query", plan.attention_bytes),
        (
            "decoder session appended key/value row",
            key_value_row_bytes,
        ),
        ("decoder session compact key/value cache", plan.cache_bytes),
        ("decoder session attention output", plan.attention_bytes),
        ("decoder session split partials", plan.split_partials_bytes),
    ] {
        runtime.validate_storage_buffer_bytes(label, bytes)?;
    }
    let cache_readback_bytes = plan.cache_bytes.checked_mul(2).ok_or_else(|| {
        RuntimeError::new(
            RuntimeErrorCode::InvalidInvocation,
            None,
            "decoder KV cache readback byte size overflowed",
        )
    })?;
    if plan.attention_bytes > runtime.capabilities.max_buffer_size
        || cache_readback_bytes > runtime.capabilities.max_buffer_size
    {
        return Err(RuntimeError::new(
            RuntimeErrorCode::Validation,
            None,
            "decoder KV session readback exceeds max_buffer_size",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn create_session<'runtime>(
    runtime: &'runtime NativeRuntime,
    descriptor: &DecoderKvSessionDescriptor<'_>,
    plan: DecoderKvSessionPlan,
    sources: &BTreeMap<KernelId, String>,
    shader_overrides: &BTreeMap<KernelId, String>,
) -> Result<NativeDecoderKvSession<'runtime>, RuntimeError> {
    const KERNELS: [KernelId; 3] = [
        KernelId::DecoderKvAppendF32,
        KernelId::DecoderGqaSplitPartialF32,
        KernelId::DecoderGqaSplitMergeF32,
    ];
    let mut pipelines = BTreeMap::new();
    for kernel in KERNELS {
        let source = sources
            .get(&kernel)
            .ok_or_else(|| RuntimeError::operation("missing decoder KV session source"))?;
        let cached = if shader_overrides.contains_key(&kernel) {
            None
        } else {
            Some(kernel)
        };
        let (pipeline, created) =
            runtime.pipeline_with_creation_status(kernel.as_str(), source, "main", cached);
        if created {
            runtime.pipeline_creations.fetch_add(1, Ordering::Relaxed);
        }
        pipelines.insert(kernel, pipeline);
    }

    let key_value_bytes = u64::try_from(plan.key_value_width)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| RuntimeError::operation("decoder KV row byte size overflowed"))?;
    let query_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-query",
        plan.attention_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    ));
    let appended_key_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-appended-key",
        key_value_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    ));
    let appended_value_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-appended-value",
        key_value_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    ));
    let key_cache_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-key-cache",
        plan.cache_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    ));
    let value_cache_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-value-cache",
        plan.cache_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    ));
    let attention_output_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-attention-output",
        plan.attention_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    ));
    let append_uniform_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-append-uniform",
        16,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    ));
    let split_partials_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-split-partials",
        plan.split_partials_bytes,
        wgpu::BufferUsages::STORAGE,
    ));
    let split_partial_uniform_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-split-partial-uniform",
        16,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    ));
    let split_merge_uniform_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-split-merge-uniform",
        16,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    ));
    let attention_readback_buffer = Box::new(runtime.create_buffer(
        "decoder-kv-session-attention-readback",
        plan.attention_bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    ));

    runtime.write_buffer(
        "decoder-kv-session-initial-key-cache",
        key_cache_buffer.as_ref(),
        0,
        bytemuck::cast_slice(descriptor.key_cache),
    );
    runtime.write_buffer(
        "decoder-kv-session-initial-value-cache",
        value_cache_buffer.as_ref(),
        0,
        bytemuck::cast_slice(descriptor.value_cache),
    );

    let append_pipeline = pipelines
        .remove(&KernelId::DecoderKvAppendF32)
        .ok_or_else(|| RuntimeError::operation("missing decoder append pipeline"))?;
    let split_partial_pipeline = pipelines
        .remove(&KernelId::DecoderGqaSplitPartialF32)
        .ok_or_else(|| RuntimeError::operation("missing decoder split partial pipeline"))?;
    let split_merge_pipeline = pipelines
        .remove(&KernelId::DecoderGqaSplitMergeF32)
        .ok_or_else(|| RuntimeError::operation("missing decoder split merge pipeline"))?;
    let append_bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoder-kv-session-append-bind-group"),
            layout: &append_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: appended_key_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: appended_value_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: key_cache_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: value_cache_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: append_uniform_buffer.as_ref().as_entire_binding(),
                },
            ],
        });
    let split_partial_bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoder-kv-session-split-partial-bind-group"),
            layout: &split_partial_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: key_cache_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: value_cache_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: split_partials_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: split_partial_uniform_buffer.as_ref().as_entire_binding(),
                },
            ],
        });
    // The split merge shader declares the cache bindings but never reads
    // them, so the derived native bind group layout only covers the
    // statically used partials, output, and uniform entries.
    let split_merge_bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoder-kv-session-split-merge-bind-group"),
            layout: &split_merge_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: split_partials_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: attention_output_buffer.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: split_merge_uniform_buffer.as_ref().as_entire_binding(),
                },
            ],
        });
    runtime.bind_group_creations.fetch_add(3, Ordering::Relaxed);

    let buffers = [
        decoder_buffer_evidence(DecoderCachedGqaBufferRole::Query, query_buffer.as_ref()),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::AppendedKey,
            appended_key_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::AppendedValue,
            appended_value_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::KeyCache,
            key_cache_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::ValueCache,
            value_cache_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::AttentionOutput,
            attention_output_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::AppendUniform,
            append_uniform_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::SplitPartials,
            split_partials_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::SplitPartialUniform,
            split_partial_uniform_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::SplitMergeUniform,
            split_merge_uniform_buffer.as_ref(),
        ),
        decoder_buffer_evidence(
            DecoderCachedGqaBufferRole::Readback,
            attention_readback_buffer.as_ref(),
        ),
    ]
    .into_iter()
    .collect();

    let append_bindings = [
        decoder_binding_evidence(0, appended_key_buffer.as_ref()),
        decoder_binding_evidence(1, appended_value_buffer.as_ref()),
        decoder_binding_evidence(2, key_cache_buffer.as_ref()),
        decoder_binding_evidence(3, value_cache_buffer.as_ref()),
        decoder_binding_evidence(4, append_uniform_buffer.as_ref()),
    ]
    .into_iter()
    .collect();
    let split_partial_bindings = [
        decoder_binding_evidence(0, query_buffer.as_ref()),
        decoder_binding_evidence(1, key_cache_buffer.as_ref()),
        decoder_binding_evidence(2, value_cache_buffer.as_ref()),
        decoder_binding_evidence(3, split_partials_buffer.as_ref()),
        decoder_binding_evidence(4, split_partial_uniform_buffer.as_ref()),
    ]
    .into_iter()
    .collect();
    let split_merge_bindings = [
        decoder_binding_evidence(0, split_partials_buffer.as_ref()),
        decoder_binding_evidence(3, attention_output_buffer.as_ref()),
        decoder_binding_evidence(4, split_merge_uniform_buffer.as_ref()),
    ]
    .into_iter()
    .collect();
    let bind_groups = [
        DecoderCachedGqaBindGroupEvidence {
            stage: DecoderCachedGqaStage::AppendKeyValue,
            bindings: append_bindings,
        },
        DecoderCachedGqaBindGroupEvidence {
            stage: DecoderCachedGqaStage::SplitGqaPartial,
            bindings: split_partial_bindings,
        },
        DecoderCachedGqaBindGroupEvidence {
            stage: DecoderCachedGqaStage::SplitGqaMerge,
            bindings: split_merge_bindings,
        },
    ]
    .into_iter()
    .collect();

    let mut shader_blake3 = BTreeMap::new();
    for kernel in KERNELS {
        let source = sources
            .get(&kernel)
            .ok_or_else(|| RuntimeError::operation("missing decoder KV session source"))?;
        shader_blake3.insert(kernel, *blake3::hash(source.as_bytes()).as_bytes());
    }
    let creation_diagnostics = DecoderKvSessionCreationDiagnostics {
        initial_cache_tokens: plan.initial_cache_tokens,
        cache_capacity: plan.cache_capacity,
        checked_error_scopes: CHECKED_SCOPE_ORDER,
        captured_errors: Vec::new(),
        shader_blake3,
        buffers,
        bind_groups,
    };

    Ok(NativeDecoderKvSession {
        runtime,
        plan,
        state: DecoderKvSessionState::Healthy,
        cache_tokens: plan.initial_cache_tokens,
        query_buffer,
        appended_key_buffer,
        appended_value_buffer,
        key_cache_buffer,
        value_cache_buffer,
        attention_output_buffer,
        append_uniform_buffer,
        _split_partials_buffer: split_partials_buffer,
        split_partial_uniform_buffer,
        split_merge_uniform_buffer,
        attention_readback_buffer,
        append_pipeline,
        split_partial_pipeline,
        split_merge_pipeline,
        append_bind_group,
        split_partial_bind_group,
        split_merge_bind_group,
        creation_diagnostics,
    })
}

impl NativeDecoderKvSession<'_> {
    pub const fn cache_tokens(&self) -> u32 {
        self.cache_tokens
    }

    pub const fn cache_capacity(&self) -> u32 {
        self.plan.cache_capacity
    }

    pub const fn creation_diagnostics(&self) -> &DecoderKvSessionCreationDiagnostics {
        &self.creation_diagnostics
    }

    pub fn step(
        &mut self,
        step: &DecoderKvSessionStep<'_>,
    ) -> Result<DecoderKvSessionStepExecution, RuntimeError> {
        match &self.state {
            DecoderKvSessionState::Healthy => {}
            DecoderKvSessionState::Poisoned => return Err(poisoned_error()),
        }
        let step_plan = self
            .plan
            .plan_step(self.cache_tokens, step)
            .map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
            })?;

        let _execution = self
            .runtime
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.state = DecoderKvSessionState::Poisoned;
        let mut scopes =
            WgpuScopeDriver::new(self.runtime.device.clone(), self.runtime.observer.clone());
        let result = drive_error_scopes(&mut scopes, || self.execute_step(step, step_plan))
            .map_err(|error| error.with_context("decoder KV session step"));
        if result.is_ok() {
            self.cache_tokens = step_plan.cache_tokens_after;
            self.state = DecoderKvSessionState::Healthy;
        }
        result
    }

    pub fn finish(self) -> Result<DecoderKvSessionSnapshot, RuntimeError> {
        match &self.state {
            DecoderKvSessionState::Healthy => {}
            DecoderKvSessionState::Poisoned => return Err(poisoned_error()),
        }
        let _execution = self
            .runtime
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut scopes =
            WgpuScopeDriver::new(self.runtime.device.clone(), self.runtime.observer.clone());
        drive_error_scopes(&mut scopes, || self.execute_finish())
            .map_err(|error| error.with_context("decoder KV session finish"))
    }

    #[allow(clippy::too_many_lines)]
    fn execute_step(
        &self,
        step: &DecoderKvSessionStep<'_>,
        step_plan: DecoderKvSessionStepPlan,
    ) -> Result<DecoderKvSessionStepExecution, RuntimeError> {
        let mut effects = Vec::with_capacity(12);
        self.write_step_operand(
            "decoder-kv-session-step-query",
            DecoderCachedGqaBufferRole::Query,
            self.query_buffer.as_ref(),
            bytemuck::cast_slice(step.query),
            &mut effects,
        );
        self.write_step_operand(
            "decoder-kv-session-step-appended-key",
            DecoderCachedGqaBufferRole::AppendedKey,
            self.appended_key_buffer.as_ref(),
            bytemuck::cast_slice(step.appended_key),
            &mut effects,
        );
        self.write_step_operand(
            "decoder-kv-session-step-appended-value",
            DecoderCachedGqaBufferRole::AppendedValue,
            self.appended_value_buffer.as_ref(),
            bytemuck::cast_slice(step.appended_value),
            &mut effects,
        );
        self.write_step_operand(
            "decoder-kv-session-step-append-uniform",
            DecoderCachedGqaBufferRole::AppendUniform,
            self.append_uniform_buffer.as_ref(),
            bytemuck::cast_slice(&step_plan.append.uniform_words),
            &mut effects,
        );
        self.write_step_operand(
            "decoder-kv-session-step-split-partial-uniform",
            DecoderCachedGqaBufferRole::SplitPartialUniform,
            self.split_partial_uniform_buffer.as_ref(),
            bytemuck::cast_slice(&step_plan.split_gqa.uniform_words[0]),
            &mut effects,
        );
        self.write_step_operand(
            "decoder-kv-session-step-split-merge-uniform",
            DecoderCachedGqaBufferRole::SplitMergeUniform,
            self.split_merge_uniform_buffer.as_ref(),
            bytemuck::cast_slice(&step_plan.split_gqa.uniform_words[1]),
            &mut effects,
        );

        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("decoder-kv-session-step-encoder"),
                });
        self.runtime
            .command_encoder_creations
            .fetch_add(1, Ordering::Relaxed);
        self.runtime
            .observe(RuntimeEvent::DecoderCommandEncoderCreated {
                label: "decoder-kv-session-step-encoder".to_owned(),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decoder-kv-session-append-pass"),
                timestamp_writes: None,
            });
            self.runtime
                .observe(RuntimeEvent::DecoderComputePassEncoded {
                    pass_index: 1,
                    stage: DecoderCachedGqaStage::AppendKeyValue,
                });
            pass.set_pipeline(&self.append_pipeline);
            pass.set_bind_group(0, &self.append_bind_group, &[]);
            pass.dispatch_workgroups(
                step_plan.append.invocation.dispatch[0],
                step_plan.append.invocation.dispatch[1],
                step_plan.append.invocation.dispatch[2],
            );
        }
        self.record_dispatch(
            7,
            DecoderCachedGqaStage::AppendKeyValue,
            step_plan.append.invocation,
            &mut effects,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decoder-kv-session-split-partial-pass"),
                timestamp_writes: None,
            });
            self.runtime
                .observe(RuntimeEvent::DecoderComputePassEncoded {
                    pass_index: 2,
                    stage: DecoderCachedGqaStage::SplitGqaPartial,
                });
            pass.set_pipeline(&self.split_partial_pipeline);
            pass.set_bind_group(0, &self.split_partial_bind_group, &[]);
            pass.dispatch_workgroups(
                step_plan.split_gqa.partial_invocation.dispatch[0],
                step_plan.split_gqa.partial_invocation.dispatch[1],
                step_plan.split_gqa.partial_invocation.dispatch[2],
            );
        }
        self.record_dispatch(
            8,
            DecoderCachedGqaStage::SplitGqaPartial,
            step_plan.split_gqa.partial_invocation,
            &mut effects,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decoder-kv-session-split-merge-pass"),
                timestamp_writes: None,
            });
            self.runtime
                .observe(RuntimeEvent::DecoderComputePassEncoded {
                    pass_index: 3,
                    stage: DecoderCachedGqaStage::SplitGqaMerge,
                });
            pass.set_pipeline(&self.split_merge_pipeline);
            pass.set_bind_group(0, &self.split_merge_bind_group, &[]);
            pass.dispatch_workgroups(
                step_plan.split_gqa.merge_invocation.dispatch[0],
                step_plan.split_gqa.merge_invocation.dispatch[1],
                step_plan.split_gqa.merge_invocation.dispatch[2],
            );
        }
        self.record_dispatch(
            9,
            DecoderCachedGqaStage::SplitGqaMerge,
            step_plan.split_gqa.merge_invocation,
            &mut effects,
        );

        encoder.copy_buffer_to_buffer(
            self.attention_output_buffer.as_ref(),
            0,
            self.attention_readback_buffer.as_ref(),
            0,
            self.plan.attention_bytes,
        );
        self.runtime
            .buffer_copy_encodings
            .fetch_add(1, Ordering::Relaxed);
        self.runtime
            .observe(RuntimeEvent::DecoderBufferCopyEncoded {
                ordinal: 10,
                source_buffer_identity: buffer_identity(self.attention_output_buffer.as_ref()),
                source_offset: 0,
                destination_buffer_identity: buffer_identity(
                    self.attention_readback_buffer.as_ref(),
                ),
                destination_offset: 0,
                byte_length: self.plan.attention_bytes,
            });
        effects.push(DecoderKvSessionEffect::CopyAttention {
            ordinal: 10,
            source_buffer_identity: buffer_identity(self.attention_output_buffer.as_ref()),
            destination_buffer_identity: buffer_identity(self.attention_readback_buffer.as_ref()),
            byte_length: self.plan.attention_bytes,
        });

        let started = Instant::now();
        let submission_index = self.runtime.submit_command_buffers([encoder.finish()]);
        effects.push(DecoderKvSessionEffect::Submit {
            ordinal: 11,
            command_buffer_count: 1,
        });
        let receiver = map_read(self.attention_readback_buffer.as_ref());
        self.runtime.map_requests.fetch_add(1, Ordering::Relaxed);
        self.runtime.observe(RuntimeEvent::DecoderMapRequested {
            buffer_identity: buffer_identity(self.attention_readback_buffer.as_ref()),
            byte_offset: 0,
            byte_length: self.plan.attention_bytes,
        });
        effects.push(DecoderKvSessionEffect::MapAttention {
            ordinal: 12,
            buffer_identity: buffer_identity(self.attention_readback_buffer.as_ref()),
            byte_length: self.plan.attention_bytes,
        });
        self.runtime
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|error| RuntimeError::operation(error.to_string()))?;
        await_mapping(receiver, "decoder KV session attention readback")?;
        let queue_wall_time_ns = elapsed_ns(started.elapsed());
        let attention = read_f32_buffer(
            self.attention_readback_buffer.as_ref(),
            self.plan.query_elements,
        )?;
        if attention.len() != self.plan.query_elements {
            return Err(RuntimeError::operation(
                "decoder KV attention readback length mismatch",
            ));
        }

        Ok(DecoderKvSessionStepExecution {
            attention,
            cache_tokens: step_plan.cache_tokens_after,
            diagnostics: DecoderKvSessionStepDiagnostics {
                cache_tokens_before: step_plan.cache_tokens_before,
                cache_tokens_after: step_plan.cache_tokens_after,
                checked_error_scopes: CHECKED_SCOPE_ORDER,
                captured_errors: Vec::new(),
                queue_wall_time_ns,
                shader_blake3: self.creation_diagnostics.shader_blake3.clone(),
                dispatch_count: 3,
                compute_pass_count: 3,
                command_buffer_count: 1,
                copy_count: 1,
                submission_count: 1,
                map_count: 1,
                readback_bytes: self.plan.attention_bytes,
                effects,
            },
        })
    }

    fn write_step_operand(
        &self,
        label: &str,
        role: DecoderCachedGqaBufferRole,
        buffer: &wgpu::Buffer,
        bytes: &[u8],
        effects: &mut Vec<DecoderKvSessionEffect>,
    ) {
        self.runtime.write_buffer(label, buffer, 0, bytes);
        effects.push(DecoderKvSessionEffect::QueueWrite {
            ordinal: effects.len() + 1,
            role,
            buffer_identity: buffer_identity(buffer),
            byte_offset: 0,
            byte_length: bytes.len() as u64,
        });
    }

    fn record_dispatch(
        &self,
        ordinal: usize,
        stage: DecoderCachedGqaStage,
        invocation: InvocationPlan,
        effects: &mut Vec<DecoderKvSessionEffect>,
    ) {
        self.runtime
            .dispatch_encodings
            .fetch_add(1, Ordering::Relaxed);
        self.runtime.observe(RuntimeEvent::DecoderDispatchEncoded {
            ordinal,
            stage,
            kernel: invocation.kernel,
            workgroups: invocation.dispatch,
        });
        effects.push(DecoderKvSessionEffect::Dispatch {
            ordinal,
            stage,
            kernel: invocation.kernel,
            workgroups: invocation.dispatch,
        });
    }

    fn execute_finish(&self) -> Result<DecoderKvSessionSnapshot, RuntimeError> {
        let readback_bytes = self
            .plan
            .cache_bytes
            .checked_mul(2)
            .ok_or_else(|| RuntimeError::operation("decoder KV finish readback overflowed"))?;
        let readback_elements = self
            .plan
            .cache_elements
            .checked_mul(2)
            .ok_or_else(|| RuntimeError::operation("decoder KV finish readback overflowed"))?;
        let readback = Box::new(self.runtime.create_buffer(
            "decoder-kv-session-finish-readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        ));
        let readback_identity = buffer_identity(readback.as_ref());
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("decoder-kv-session-finish-encoder"),
                });
        self.runtime
            .command_encoder_creations
            .fetch_add(1, Ordering::Relaxed);
        self.runtime
            .observe(RuntimeEvent::DecoderCommandEncoderCreated {
                label: "decoder-kv-session-finish-encoder".to_owned(),
            });
        encoder.copy_buffer_to_buffer(
            self.key_cache_buffer.as_ref(),
            0,
            readback.as_ref(),
            0,
            self.plan.cache_bytes,
        );
        self.runtime
            .buffer_copy_encodings
            .fetch_add(1, Ordering::Relaxed);
        self.runtime
            .observe(RuntimeEvent::DecoderBufferCopyEncoded {
                ordinal: 1,
                source_buffer_identity: buffer_identity(self.key_cache_buffer.as_ref()),
                source_offset: 0,
                destination_buffer_identity: readback_identity,
                destination_offset: 0,
                byte_length: self.plan.cache_bytes,
            });
        encoder.copy_buffer_to_buffer(
            self.value_cache_buffer.as_ref(),
            0,
            readback.as_ref(),
            self.plan.cache_bytes,
            self.plan.cache_bytes,
        );
        self.runtime
            .buffer_copy_encodings
            .fetch_add(1, Ordering::Relaxed);
        self.runtime
            .observe(RuntimeEvent::DecoderBufferCopyEncoded {
                ordinal: 2,
                source_buffer_identity: buffer_identity(self.value_cache_buffer.as_ref()),
                source_offset: 0,
                destination_buffer_identity: readback_identity,
                destination_offset: self.plan.cache_bytes,
                byte_length: self.plan.cache_bytes,
            });

        let started = Instant::now();
        let submission_index = self.runtime.submit_command_buffers([encoder.finish()]);
        let receiver = map_read(readback.as_ref());
        self.runtime.map_requests.fetch_add(1, Ordering::Relaxed);
        self.runtime.observe(RuntimeEvent::DecoderMapRequested {
            buffer_identity: readback_identity,
            byte_offset: 0,
            byte_length: readback_bytes,
        });
        self.runtime
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|error| RuntimeError::operation(error.to_string()))?;
        await_mapping(receiver, "decoder KV session cache readback")?;
        let queue_wall_time_ns = elapsed_ns(started.elapsed());
        let values = read_f32_buffer(readback.as_ref(), readback_elements)?;
        if values.len() != readback_elements {
            return Err(RuntimeError::operation(
                "decoder KV finish readback length mismatch",
            ));
        }
        let key_cache = values[..self.plan.cache_elements].to_vec();
        let value_cache = values[self.plan.cache_elements..].to_vec();
        Ok(DecoderKvSessionSnapshot {
            key_cache,
            value_cache,
            cache_tokens: self.cache_tokens,
            cache_capacity: self.plan.cache_capacity,
            diagnostics: DecoderKvSessionSnapshotDiagnostics {
                checked_error_scopes: CHECKED_SCOPE_ORDER,
                captured_errors: Vec::new(),
                queue_wall_time_ns,
                readback_buffer_identity: readback_identity,
                readback_bytes,
                copy_count: 2,
                command_buffer_count: 1,
                submission_count: 1,
                map_count: 1,
            },
        })
    }
}

fn poisoned_error() -> RuntimeError {
    RuntimeError::operation("decoder KV session is terminally poisoned")
}
