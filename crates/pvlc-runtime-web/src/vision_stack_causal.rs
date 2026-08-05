use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt,
    future::Future,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    AsyncSessionLease, AsyncSessionOwner, CompletionAction, CompletionOutcome, SessionOwnerError,
};

pub(crate) const VISION_STACK_STREAMING_WEIGHT_SLOTS: usize = 16;
pub(crate) const VISION_STACK_POST_NORM_WEIGHT_SLOTS: usize = 2;

pub(crate) struct VisionStackStreamingWeightRange {
    offset_bytes: u64,
    length_bytes: u64,
}

impl Copy for VisionStackStreamingWeightRange {}

impl Clone for VisionStackStreamingWeightRange {
    fn clone(&self) -> Self {
        *self
    }
}

impl VisionStackStreamingWeightRange {
    pub(crate) const fn new(offset_bytes: u64, length_bytes: u64) -> Self {
        Self {
            offset_bytes,
            length_bytes,
        }
    }

    pub(crate) const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub(crate) const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }
}

pub(crate) struct VisionStackStreamingLayerSchedule {
    layer_index: u32,
    checkpoint_slot: Option<usize>,
    ranges: [VisionStackStreamingWeightRange; VISION_STACK_STREAMING_WEIGHT_SLOTS],
}

impl VisionStackStreamingLayerSchedule {
    pub(crate) const fn new(
        layer_index: u32,
        checkpoint_slot: Option<usize>,
        ranges: [VisionStackStreamingWeightRange; VISION_STACK_STREAMING_WEIGHT_SLOTS],
    ) -> Self {
        Self {
            layer_index,
            checkpoint_slot,
            ranges,
        }
    }
}

pub(crate) struct VisionStackStreamingWeightCache<Resource> {
    lengths: Option<[u64; VISION_STACK_STREAMING_WEIGHT_SLOTS]>,
    resources: Option<Vec<Resource>>,
}

impl<Resource> VisionStackStreamingWeightCache<Resource> {
    pub(crate) const fn new() -> Self {
        Self {
            lengths: None,
            resources: None,
        }
    }

    pub(crate) fn resources(&self) -> Option<&[Resource]> {
        self.resources.as_deref()
    }
}

impl<Resource> Default for VisionStackStreamingWeightCache<Resource> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VisionStackResidentCacheDisposition {
    Cold,
    Ready,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VisionStackResidentCacheError {
    message: String,
}

impl VisionStackResidentCacheError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for VisionStackResidentCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VisionStackResidentCacheError {}

pub(crate) struct VisionStackResidentWeightCache<Key, Resource> {
    key: Option<Key>,
    layer_count: usize,
    layers: Vec<Option<Vec<Resource>>>,
    loaded_layers: usize,
    post_norm: Option<Vec<Resource>>,
    ready: bool,
}

impl<Key, Resource> VisionStackResidentWeightCache<Key, Resource> {
    pub(crate) const fn new() -> Self {
        Self {
            key: None,
            layer_count: 0,
            layers: Vec::new(),
            loaded_layers: 0,
            post_norm: None,
            ready: false,
        }
    }

    #[allow(dead_code)]
    pub(crate) const fn loaded_layer_count(&self) -> usize {
        self.loaded_layers
    }

    #[allow(dead_code)]
    pub(crate) fn clear(&mut self) {
        self.key = None;
        self.layer_count = 0;
        self.layers.clear();
        self.loaded_layers = 0;
        self.post_norm = None;
        self.ready = false;
    }

    pub(crate) fn store_layer(
        &mut self,
        layer: usize,
        resources: Vec<Resource>,
    ) -> Result<(), VisionStackResidentCacheError> {
        if self.key.is_none() || self.layer_count == 0 {
            return Err(VisionStackResidentCacheError::new(
                "resident vision-weight cache was not prepared",
            ));
        }
        if self.ready {
            return Err(VisionStackResidentCacheError::new(
                "resident vision-weight cache is already complete",
            ));
        }
        if layer != self.loaded_layers {
            return Err(VisionStackResidentCacheError::new(format!(
                "resident vision layer {layer} arrived out of order; expected {}",
                self.loaded_layers
            )));
        }
        if resources.len() != VISION_STACK_STREAMING_WEIGHT_SLOTS {
            return Err(VisionStackResidentCacheError::new(format!(
                "resident vision layer {layer} has {} resources; expected {VISION_STACK_STREAMING_WEIGHT_SLOTS}",
                resources.len()
            )));
        }
        let slot = self.layers.get_mut(layer).ok_or_else(|| {
            VisionStackResidentCacheError::new(format!(
                "resident vision layer {layer} exceeds configured layer count {}",
                self.layer_count
            ))
        })?;
        if slot.is_some() {
            return Err(VisionStackResidentCacheError::new(format!(
                "resident vision layer {layer} is already populated"
            )));
        }
        *slot = Some(resources);
        self.loaded_layers += 1;
        Ok(())
    }

    pub(crate) fn store_post_norm(
        &mut self,
        resources: Vec<Resource>,
    ) -> Result<(), VisionStackResidentCacheError> {
        if self.key.is_none() || self.layer_count == 0 {
            return Err(VisionStackResidentCacheError::new(
                "resident vision-weight cache was not prepared",
            ));
        }
        if self.loaded_layers != self.layer_count {
            return Err(VisionStackResidentCacheError::new(format!(
                "resident post-norm arrived after {} of {} layers",
                self.loaded_layers, self.layer_count
            )));
        }
        if resources.len() != VISION_STACK_POST_NORM_WEIGHT_SLOTS {
            return Err(VisionStackResidentCacheError::new(format!(
                "resident post-norm has {} resources; expected {VISION_STACK_POST_NORM_WEIGHT_SLOTS}",
                resources.len()
            )));
        }
        if self.post_norm.is_some() || self.ready {
            return Err(VisionStackResidentCacheError::new(
                "resident post-norm is already populated",
            ));
        }
        self.post_norm = Some(resources);
        self.ready = true;
        Ok(())
    }
}

impl<Key: Eq, Resource> VisionStackResidentWeightCache<Key, Resource> {
    pub(crate) fn prepare(
        &mut self,
        key: Key,
        layer_count: usize,
    ) -> Result<VisionStackResidentCacheDisposition, VisionStackResidentCacheError> {
        if layer_count == 0 {
            return Err(VisionStackResidentCacheError::new(
                "resident vision-weight cache requires at least one layer",
            ));
        }
        if self.ready
            && self.layer_count == layer_count
            && self.key.as_ref().is_some_and(|stored| stored == &key)
        {
            return Ok(VisionStackResidentCacheDisposition::Ready);
        }
        self.key = Some(key);
        self.layer_count = layer_count;
        self.layers = (0..layer_count).map(|_| None).collect();
        self.loaded_layers = 0;
        self.post_norm = None;
        self.ready = false;
        Ok(VisionStackResidentCacheDisposition::Cold)
    }

    pub(crate) fn is_ready_for(&self, key: &Key, layer_count: usize) -> bool {
        self.ready
            && self.layer_count == layer_count
            && self.key.as_ref().is_some_and(|stored| stored == key)
    }

    pub(crate) fn is_prepared_for(&self, key: &Key, layer_count: usize) -> bool {
        self.layer_count == layer_count && self.key.as_ref().is_some_and(|stored| stored == key)
    }
}

impl<Key, Resource: Clone> VisionStackResidentWeightCache<Key, Resource> {
    pub(crate) fn clone_layer(
        &self,
        layer: usize,
    ) -> Result<Vec<Resource>, VisionStackResidentCacheError> {
        if !self.ready {
            return Err(VisionStackResidentCacheError::new(
                "resident vision-weight cache is not complete",
            ));
        }
        self.layers
            .get(layer)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or_else(|| {
                VisionStackResidentCacheError::new(format!(
                    "resident vision layer {layer} is unavailable"
                ))
            })
    }

    pub(crate) fn clone_post_norm(&self) -> Result<Vec<Resource>, VisionStackResidentCacheError> {
        if !self.ready {
            return Err(VisionStackResidentCacheError::new(
                "resident vision-weight cache is not complete",
            ));
        }
        self.post_norm
            .clone()
            .ok_or_else(|| VisionStackResidentCacheError::new("resident post-norm is unavailable"))
    }
}

impl<Key, Resource> Default for VisionStackResidentWeightCache<Key, Resource> {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) enum VisionStackResidentFailure<Error> {
    Admission(Error),
    Effect {
        error: Error,
        boundary: VisionStackGpuEffectBoundary,
    },
    Cache(VisionStackResidentCacheError),
}

impl<Error: fmt::Debug> fmt::Debug for VisionStackResidentFailure<Error> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => formatter.debug_tuple("Admission").field(error).finish(),
            Self::Effect { error, boundary } => formatter
                .debug_struct("Effect")
                .field("error", error)
                .field("boundary", boundary)
                .finish(),
            Self::Cache(error) => formatter.debug_tuple("Cache").field(error).finish(),
        }
    }
}

pub(crate) enum VisionStackStreamingFailure<Error> {
    Unavailable(SessionOwnerError),
    Admission(Error),
    CacheLengthMismatch {
        slot: usize,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    Effect {
        error: Error,
        boundary: VisionStackGpuEffectBoundary,
    },
    Completion(CompletionOutcome),
}

impl<Error: fmt::Debug> fmt::Debug for VisionStackStreamingFailure<Error> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(error) => formatter.debug_tuple("Unavailable").field(error).finish(),
            Self::Admission(error) => formatter.debug_tuple("Admission").field(error).finish(),
            Self::CacheLengthMismatch {
                slot,
                expected_bytes,
                actual_bytes,
            } => formatter
                .debug_struct("CacheLengthMismatch")
                .field("slot", slot)
                .field("expected_bytes", expected_bytes)
                .field("actual_bytes", actual_bytes)
                .finish(),
            Self::Effect { error, boundary } => formatter
                .debug_struct("Effect")
                .field("error", error)
                .field("boundary", boundary)
                .finish(),
            Self::Completion(outcome) => {
                formatter.debug_tuple("Completion").field(outcome).finish()
            }
        }
    }
}

impl<Error: PartialEq> PartialEq for VisionStackStreamingFailure<Error> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Unavailable(left), Self::Unavailable(right)) => left == right,
            (Self::Admission(left), Self::Admission(right)) => left == right,
            (
                Self::CacheLengthMismatch {
                    slot: left_slot,
                    expected_bytes: left_expected,
                    actual_bytes: left_actual,
                },
                Self::CacheLengthMismatch {
                    slot: right_slot,
                    expected_bytes: right_expected,
                    actual_bytes: right_actual,
                },
            ) => {
                left_slot == right_slot
                    && left_expected == right_expected
                    && left_actual == right_actual
            }
            (
                Self::Effect {
                    error: left_error,
                    boundary: left_boundary,
                },
                Self::Effect {
                    error: right_error,
                    boundary: right_boundary,
                },
            ) => left_error == right_error && left_boundary == right_boundary,
            (Self::Completion(left), Self::Completion(right)) => left == right,
            _ => false,
        }
    }
}

impl<Error: Eq> Eq for VisionStackStreamingFailure<Error> {}

pub(crate) fn run_vision_stack_streaming_layer<
    Session,
    Resource,
    Error,
    Validate,
    Allocate,
    Upload,
    Submit,
>(
    cache: &mut VisionStackStreamingWeightCache<Resource>,
    session: &mut Session,
    validate: Validate,
    mut allocate: Allocate,
    mut upload: Upload,
    submit: Submit,
) -> Result<(), VisionStackStreamingFailure<Error>>
where
    Session: Clone,
    Validate: FnOnce(&mut Session) -> Result<VisionStackStreamingLayerSchedule, Error>,
    Allocate: FnMut(usize, VisionStackStreamingWeightRange) -> Result<Resource, Error>,
    Upload: FnMut(usize, VisionStackStreamingWeightRange, &Resource) -> Result<(), Error>,
    Submit: FnOnce(&mut Session, u32, Option<usize>, &[Resource]) -> Result<(), Error>,
{
    let mut shadow = session.clone();
    let schedule = validate(&mut shadow).map_err(VisionStackStreamingFailure::Admission)?;
    if let Some(lengths) = cache.lengths {
        for (slot, (expected_bytes, range)) in lengths.into_iter().zip(schedule.ranges).enumerate()
        {
            let actual_bytes = range.length_bytes();
            if expected_bytes != actual_bytes {
                return Err(VisionStackStreamingFailure::CacheLengthMismatch {
                    slot,
                    expected_bytes,
                    actual_bytes,
                });
            }
        }
    }

    if cache.resources.is_none() {
        let mut resources = Vec::with_capacity(VISION_STACK_STREAMING_WEIGHT_SLOTS);
        for (slot, range) in schedule.ranges.iter().copied().enumerate() {
            let resource =
                allocate(slot, range).map_err(|error| VisionStackStreamingFailure::Effect {
                    error,
                    boundary: VisionStackGpuEffectBoundary::PostEffect,
                })?;
            resources.push(resource);
        }
        enforce_causal_invariant(
            resources.len() == VISION_STACK_STREAMING_WEIGHT_SLOTS,
            "vision-stack streaming allocation did not create one exact weight set",
        );
        cache.lengths = Some(schedule.ranges.map(|range| range.length_bytes()));
        cache.resources = Some(resources);
    }

    let resources = cache
        .resources()
        .expect("vision-stack streaming cache was initialized");
    enforce_causal_invariant(
        resources.len() == VISION_STACK_STREAMING_WEIGHT_SLOTS,
        "vision-stack streaming cache contains an invalid resource count",
    );
    for (slot, (range, resource)) in schedule.ranges.iter().copied().zip(resources).enumerate() {
        upload(slot, range, resource).map_err(|error| VisionStackStreamingFailure::Effect {
            error,
            boundary: VisionStackGpuEffectBoundary::PostEffect,
        })?;
    }
    submit(
        &mut shadow,
        schedule.layer_index,
        schedule.checkpoint_slot,
        resources,
    )
    .map_err(|error| VisionStackStreamingFailure::Effect {
        error,
        boundary: VisionStackGpuEffectBoundary::PostEffect,
    })?;
    *session = shadow;
    Ok(())
}

pub(crate) fn run_vision_stack_resident_cold_layer<
    Key,
    Session,
    Resource,
    Error,
    Validate,
    Allocate,
    Upload,
    Submit,
>(
    cache: &mut VisionStackResidentWeightCache<Key, Resource>,
    session: &mut Session,
    validate: Validate,
    mut allocate: Allocate,
    mut upload: Upload,
    submit: Submit,
) -> Result<(), VisionStackResidentFailure<Error>>
where
    Session: Clone,
    Validate: FnOnce(&mut Session) -> Result<VisionStackStreamingLayerSchedule, Error>,
    Allocate: FnMut(usize, VisionStackStreamingWeightRange) -> Result<Resource, Error>,
    Upload: FnMut(usize, VisionStackStreamingWeightRange, &Resource) -> Result<(), Error>,
    Submit: FnOnce(&mut Session, u32, Option<usize>, &[Resource]) -> Result<(), Error>,
{
    let mut shadow = session.clone();
    let schedule = validate(&mut shadow).map_err(VisionStackResidentFailure::Admission)?;
    let mut resources = Vec::with_capacity(VISION_STACK_STREAMING_WEIGHT_SLOTS);
    for (slot, range) in schedule.ranges.iter().copied().enumerate() {
        let resource =
            allocate(slot, range).map_err(|error| VisionStackResidentFailure::Effect {
                error,
                boundary: VisionStackGpuEffectBoundary::PostEffect,
            })?;
        resources.push(resource);
    }
    enforce_causal_invariant(
        resources.len() == VISION_STACK_STREAMING_WEIGHT_SLOTS,
        "resident vision allocation did not create one exact layer weight set",
    );
    for (slot, (range, resource)) in schedule.ranges.iter().copied().zip(&resources).enumerate() {
        upload(slot, range, resource).map_err(|error| VisionStackResidentFailure::Effect {
            error,
            boundary: VisionStackGpuEffectBoundary::PostEffect,
        })?;
    }
    submit(
        &mut shadow,
        schedule.layer_index,
        schedule.checkpoint_slot,
        &resources,
    )
    .map_err(|error| VisionStackResidentFailure::Effect {
        error,
        boundary: VisionStackGpuEffectBoundary::PostEffect,
    })?;
    let layer_index = usize::try_from(schedule.layer_index).map_err(|_| {
        VisionStackResidentFailure::Cache(VisionStackResidentCacheError::new(format!(
            "resident vision layer {} does not fit usize",
            schedule.layer_index
        )))
    })?;
    cache
        .store_layer(layer_index, resources)
        .map_err(VisionStackResidentFailure::Cache)?;
    *session = shadow;
    Ok(())
}

pub(crate) fn run_vision_stack_streaming_session_layer<
    Session,
    Resource,
    Error,
    Validate,
    Allocate,
    Upload,
    Submit,
>(
    owner: &AsyncSessionOwner<Session>,
    execution_busy: &Cell<bool>,
    cache: &mut VisionStackStreamingWeightCache<Resource>,
    validate: Validate,
    allocate: Allocate,
    upload: Upload,
    submit: Submit,
) -> Result<(), VisionStackStreamingFailure<Error>>
where
    Session: Clone,
    Validate: FnOnce(&mut Session) -> Result<VisionStackStreamingLayerSchedule, Error>,
    Allocate: FnMut(usize, VisionStackStreamingWeightRange) -> Result<Resource, Error>,
    Upload: FnMut(usize, VisionStackStreamingWeightRange, &Resource) -> Result<(), Error>,
    Submit: FnOnce(&mut Session, u32, Option<usize>, &[Resource]) -> Result<(), Error>,
{
    let (lease, mut session) = owner
        .acquire()
        .map_err(VisionStackStreamingFailure::Unavailable)?;
    let outcome =
        run_vision_stack_streaming_layer(cache, &mut session, validate, allocate, upload, submit);
    let action = match &outcome {
        Ok(())
        | Err(VisionStackStreamingFailure::Admission(_))
        | Err(VisionStackStreamingFailure::CacheLengthMismatch { .. }) => CompletionAction::Restore,
        Err(VisionStackStreamingFailure::Effect { .. })
        | Err(VisionStackStreamingFailure::Unavailable(_))
        | Err(VisionStackStreamingFailure::Completion(_)) => CompletionAction::Finish,
    };
    let completion = owner.complete(lease, session, action);
    let _ = coordinate_vision_stack_completion_busy(execution_busy, completion);
    match (&outcome, completion) {
        (Ok(()), CompletionOutcome::Restored)
        | (Err(VisionStackStreamingFailure::Admission(_)), CompletionOutcome::Restored)
        | (
            Err(VisionStackStreamingFailure::CacheLengthMismatch { .. }),
            CompletionOutcome::Restored,
        )
        | (Err(VisionStackStreamingFailure::Effect { .. }), CompletionOutcome::Finished) => outcome,
        _ => Err(VisionStackStreamingFailure::Completion(completion)),
    }
}

pub(crate) struct VisionStackErrorScopeLedger<Scope> {
    scopes: Vec<Scope>,
}

impl<Scope> VisionStackErrorScopeLedger<Scope> {
    fn new() -> Self {
        Self {
            scopes: Vec::with_capacity(3),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.scopes.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.scopes.is_empty()
    }

    pub(crate) fn scopes(&self) -> &[Scope] {
        &self.scopes
    }
}

pub(crate) struct VisionStackErrorScopeAuthority<'a, Scope> {
    healthy: &'a Cell<bool>,
    occupied: &'a Cell<bool>,
    ledger: VisionStackErrorScopeLedger<Scope>,
}

impl<'a, Scope> VisionStackErrorScopeAuthority<'a, Scope> {
    pub(crate) fn acquire<Error>(
        healthy: &'a Cell<bool>,
        occupied: &'a Cell<bool>,
        poisoned_error: Error,
        occupied_error: Error,
    ) -> Result<Self, Error> {
        require_vision_stack_error_scope_admission_available(
            healthy,
            occupied,
            poisoned_error,
            occupied_error,
        )?;
        occupied.set(true);
        Ok(Self {
            healthy,
            occupied,
            ledger: VisionStackErrorScopeLedger::new(),
        })
    }

    pub(crate) fn after_first_push(&mut self, scope: Scope) {
        enforce_causal_invariant(
            self.healthy.get(),
            "poisoned vision-stack error-scope authority cannot record a push",
        );
        enforce_causal_invariant(
            self.ledger.is_empty(),
            "first vision-stack error-scope push must initialize an empty ledger",
        );
        self.ledger.scopes.insert(0, scope);
    }

    fn record_pushed_scope(&mut self, scope: Scope) {
        enforce_causal_invariant(
            self.healthy.get(),
            "poisoned vision-stack error-scope authority cannot record a push",
        );
        self.ledger.scopes.insert(0, scope);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ledger.is_empty()
    }

    pub(crate) fn scopes(&self) -> &[Scope] {
        self.ledger.scopes()
    }
}

impl<Scope> Drop for VisionStackErrorScopeAuthority<'_, Scope> {
    fn drop(&mut self) {
        if !self.is_empty() {
            self.healthy.set(false);
        }
        self.occupied.set(false);
    }
}

pub(crate) struct VisionStackErrorScopeDrain<Value, Error> {
    popped: Vec<Value>,
    failures: Vec<Error>,
    remaining: usize,
}

impl<Value, Error> VisionStackErrorScopeDrain<Value, Error> {
    pub(crate) fn into_parts(self) -> (Vec<Value>, Vec<Error>, usize) {
        (self.popped, self.failures, self.remaining)
    }
}

pub(crate) struct VisionStackErrorScopePushFailure<Value, Error> {
    push_error: Error,
    cleanup: VisionStackErrorScopeDrain<Value, Error>,
}

impl<Value: fmt::Debug, Error: fmt::Debug> fmt::Debug
    for VisionStackErrorScopePushFailure<Value, Error>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisionStackErrorScopePushFailure")
            .field("push_error", &self.push_error)
            .field("popped", &self.cleanup.popped)
            .field("failures", &self.cleanup.failures)
            .field("remaining", &self.cleanup.remaining)
            .finish()
    }
}

impl<Value, Error> VisionStackErrorScopePushFailure<Value, Error> {
    pub(crate) fn into_parts(self) -> (Error, VisionStackErrorScopeDrain<Value, Error>) {
        (self.push_error, self.cleanup)
    }
}

pub(crate) enum VisionStackErrorScopePopAttempt<Value, Error> {
    Popped(Result<Value, Error>),
    NotPopped(Error),
}

pub(crate) fn require_vision_stack_error_scope_admission_available<Error>(
    healthy: &Cell<bool>,
    occupied: &Cell<bool>,
    poisoned_error: Error,
    occupied_error: Error,
) -> Result<(), Error> {
    if !healthy.get() {
        return Err(poisoned_error);
    }
    if occupied.get() {
        return Err(occupied_error);
    }
    Ok(())
}

pub(crate) async fn observe_vision_stack_error_scope_pop<
    Pending,
    RawValue,
    Value,
    Error,
    Wait,
    WaitFuture,
    Normalize,
>(
    invocation: Result<Pending, Error>,
    wait: Wait,
    normalize: Normalize,
) -> VisionStackErrorScopePopAttempt<Value, Error>
where
    Wait: FnOnce(Pending) -> WaitFuture,
    WaitFuture: Future<Output = Result<RawValue, Error>>,
    Normalize: FnOnce(RawValue) -> Result<Value, Error>,
{
    let pending = match invocation {
        Ok(pending) => pending,
        Err(error) => return VisionStackErrorScopePopAttempt::NotPopped(error),
    };
    let raw_value = match wait(pending).await {
        Ok(raw_value) => raw_value,
        Err(error) => return VisionStackErrorScopePopAttempt::NotPopped(error),
    };
    VisionStackErrorScopePopAttempt::Popped(normalize(raw_value))
}

pub(crate) async fn drain_vision_stack_error_scopes<Scope, Value, Error, Pop, PopFuture>(
    authority: &mut VisionStackErrorScopeAuthority<'_, Scope>,
    mut pop: Pop,
) -> VisionStackErrorScopeDrain<Value, Error>
where
    Scope: Copy,
    Pop: FnMut(Scope) -> PopFuture,
    PopFuture: Future<Output = VisionStackErrorScopePopAttempt<Value, Error>>,
{
    let mut popped = Vec::new();
    let mut failures = Vec::new();
    if authority.healthy.get() && !authority.is_empty() {
        while let Some(scope) = authority.scopes().first().copied() {
            match pop(scope).await {
                VisionStackErrorScopePopAttempt::Popped(result) => {
                    let _ = authority.ledger.scopes.remove(0);
                    match result {
                        Ok(value) => popped.push(value),
                        Err(error) => failures.push(error),
                    }
                }
                VisionStackErrorScopePopAttempt::NotPopped(error) => {
                    authority.healthy.set(false);
                    failures.push(error);
                    break;
                }
            }
        }
    }
    VisionStackErrorScopeDrain {
        popped,
        failures,
        remaining: authority.ledger.len(),
    }
}

pub(crate) async fn push_vision_stack_error_scope_or_drain<
    Scope,
    Value,
    Error,
    Push,
    Pop,
    PopFuture,
>(
    authority: &mut VisionStackErrorScopeAuthority<'_, Scope>,
    scope: Scope,
    push: Push,
    pop: Pop,
) -> Result<(), VisionStackErrorScopePushFailure<Value, Error>>
where
    Scope: Copy,
    Push: FnOnce(&Scope) -> Result<(), Error>,
    Pop: FnMut(Scope) -> PopFuture,
    PopFuture: Future<Output = VisionStackErrorScopePopAttempt<Value, Error>>,
{
    enforce_causal_invariant(
        authority.healthy.get(),
        "poisoned vision-stack error-scope authority cannot push",
    );
    match push(&scope) {
        Ok(()) => {
            authority.record_pushed_scope(scope);
            Ok(())
        }
        Err(push_error) => {
            let cleanup = drain_vision_stack_error_scopes(authority, pop).await;
            Err(VisionStackErrorScopePushFailure {
                push_error,
                cleanup,
            })
        }
    }
}

pub(crate) struct VisionStackErrorScopedOperation<T, Value, Error> {
    operation: Result<T, Error>,
    cleanup: VisionStackErrorScopeDrain<Value, Error>,
}

impl<T, Value, Error> VisionStackErrorScopedOperation<T, Value, Error> {
    pub(crate) fn into_parts(self) -> (Result<T, Error>, VisionStackErrorScopeDrain<Value, Error>) {
        (self.operation, self.cleanup)
    }
}

pub(crate) async fn run_vision_stack_error_scoped_operation<
    Scope,
    T,
    Value,
    Error,
    Operation,
    Pop,
    PopFuture,
>(
    mut authority: VisionStackErrorScopeAuthority<'_, Scope>,
    operation: Operation,
    pop: Pop,
) -> VisionStackErrorScopedOperation<T, Value, Error>
where
    Scope: Copy,
    Operation: Future<Output = Result<T, Error>>,
    Pop: FnMut(Scope) -> PopFuture,
    PopFuture: Future<Output = VisionStackErrorScopePopAttempt<Value, Error>>,
{
    let operation = operation.await;
    let cleanup = drain_vision_stack_error_scopes(&mut authority, pop).await;
    VisionStackErrorScopedOperation { operation, cleanup }
}

pub(crate) enum VisionStackGpuEffectBoundary {
    PreEffect,
    PostEffect,
}

impl Copy for VisionStackGpuEffectBoundary {}

impl Clone for VisionStackGpuEffectBoundary {
    fn clone(&self) -> Self {
        *self
    }
}

impl PartialEq for VisionStackGpuEffectBoundary {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::PreEffect, Self::PreEffect) | (Self::PostEffect, Self::PostEffect) => true,
            (Self::PreEffect, Self::PostEffect) | (Self::PostEffect, Self::PreEffect) => false,
        }
    }
}

impl Eq for VisionStackGpuEffectBoundary {}

impl fmt::Debug for VisionStackGpuEffectBoundary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreEffect => formatter.write_str("PreEffect"),
            Self::PostEffect => formatter.write_str("PostEffect"),
        }
    }
}

static NEXT_VISION_STACK_EFFECT_TRACKER_ID: AtomicU64 = AtomicU64::new(1);

struct VisionStackEffectTracker {
    id: u64,
    boundary: VisionStackGpuEffectBoundary,
}

fn enforce_causal_invariant(condition: bool, message: &'static str) {
    if !condition {
        std::panic::panic_any(message);
    }
}

impl VisionStackEffectTracker {
    fn new() -> Self {
        let id = NEXT_VISION_STACK_EFFECT_TRACKER_ID.fetch_add(1, Ordering::Relaxed);
        enforce_causal_invariant(id != 0, "vision-stack effect tracker ID overflowed");
        Self {
            id,
            boundary: VisionStackGpuEffectBoundary::PreEffect,
        }
    }

    fn mark_post_effect(&mut self) {
        enforce_causal_invariant(
            self.boundary == VisionStackGpuEffectBoundary::PreEffect,
            "vision-stack effect boundary may cross into POST exactly once",
        );
        self.boundary = VisionStackGpuEffectBoundary::PostEffect;
    }
}

pub(crate) struct VisionStackPostEffectToken {
    tracker_id: u64,
    boundary: VisionStackGpuEffectBoundary,
}

impl VisionStackPostEffectToken {
    #[must_use]
    pub(crate) const fn effect_tracker_id(&self) -> u64 {
        self.tracker_id
    }

    #[must_use]
    pub(crate) const fn effect_boundary(&self) -> VisionStackGpuEffectBoundary {
        self.boundary
    }
}

pub(crate) struct VisionStackOperationEffectResult<T, E> {
    result: Result<T, E>,
    tracker_id: u64,
    boundary: VisionStackGpuEffectBoundary,
}

impl<T, E> VisionStackOperationEffectResult<T, E> {
    fn failure(error: E, tracker_id: u64, boundary: VisionStackGpuEffectBoundary) -> Self {
        Self {
            result: Err(error),
            tracker_id,
            boundary,
        }
    }

    fn observed(
        result: Result<T, E>,
        tracker_id: u64,
        boundary: VisionStackGpuEffectBoundary,
    ) -> Self {
        Self {
            result,
            tracker_id,
            boundary,
        }
    }

    #[must_use]
    const fn effect_tracker_id(&self) -> u64 {
        self.tracker_id
    }

    #[must_use]
    pub(crate) const fn effect_boundary(&self) -> VisionStackGpuEffectBoundary {
        self.boundary
    }
}

pub(crate) struct VisionStackOperationEffectBoundary {
    tracker: VisionStackEffectTracker,
    post_effect: Option<VisionStackPostEffectToken>,
}

fn store_vision_stack_post_effect_token(
    slot: &mut Option<VisionStackPostEffectToken>,
    token: VisionStackPostEffectToken,
) -> &VisionStackPostEffectToken {
    &*slot.insert(token)
}

impl VisionStackOperationEffectBoundary {
    pub(crate) fn new() -> Self {
        Self {
            tracker: VisionStackEffectTracker::new(),
            post_effect: None,
        }
    }

    #[must_use]
    pub(crate) fn failure<T, E>(&self, error: E) -> VisionStackOperationEffectResult<T, E> {
        VisionStackOperationEffectResult::failure(error, self.tracker.id, self.tracker.boundary)
    }

    pub(crate) async fn run_webgpu_effect<'a, T, E, Effect>(
        &'a mut self,
        effect: Effect,
    ) -> VisionStackOperationEffectResult<T, E>
    where
        Effect: std::ops::AsyncFnOnce(&'a VisionStackPostEffectToken) -> Result<T, E>,
    {
        self.tracker.mark_post_effect();
        let post_effect = store_vision_stack_post_effect_token(
            &mut self.post_effect,
            VisionStackPostEffectToken {
                tracker_id: self.tracker.id,
                boundary: self.tracker.boundary,
            },
        );
        let result = effect(post_effect).await;
        VisionStackOperationEffectResult::observed(result, self.tracker.id, self.tracker.boundary)
    }
}

pub(crate) struct VisionStackOperationFailure<E> {
    error: E,
    tracker_id: u64,
    boundary: VisionStackGpuEffectBoundary,
}

impl<E: fmt::Debug> fmt::Debug for VisionStackOperationFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisionStackOperationFailure")
            .field("error", &self.error)
            .field("tracker_id", &self.tracker_id)
            .field("boundary", &self.boundary)
            .finish()
    }
}

pub(crate) struct VisionStackOperationTransaction<Session, T, E> {
    original: Session,
    shadow: Session,
    outcome: Result<T, VisionStackOperationFailure<E>>,
    tracker_id: u64,
    boundary: VisionStackGpuEffectBoundary,
}

impl<Session: Clone, T, E> VisionStackOperationTransaction<Session, T, E> {
    pub(crate) fn from_effect_result(
        session: Session,
        shadow: Session,
        effect_result: VisionStackOperationEffectResult<T, E>,
        effect_boundary: VisionStackOperationEffectBoundary,
    ) -> Self {
        let tracker_id = effect_result.effect_tracker_id();
        let boundary = effect_result.effect_boundary();
        enforce_causal_invariant(
            tracker_id == effect_boundary.tracker.id,
            "vision-stack effect result belongs to another tracker",
        );
        enforce_causal_invariant(
            boundary == effect_boundary.tracker.boundary,
            "vision-stack effect result carries a stale boundary",
        );
        let shadow = if boundary == VisionStackGpuEffectBoundary::PreEffect
            && effect_result.result.is_err()
        {
            session.clone()
        } else {
            shadow
        };
        let outcome = effect_result
            .result
            .map_err(|error| VisionStackOperationFailure {
                error,
                tracker_id,
                boundary,
            });
        Self {
            original: session,
            shadow,
            outcome,
            tracker_id,
            boundary,
        }
    }
}

pub(crate) async fn run_vision_stack_operation_transaction<
    Session,
    Validated,
    Value,
    Error,
    Validate,
    Effect,
>(
    session: Session,
    validate: Validate,
    effect: Effect,
) -> VisionStackOperationTransaction<Session, Value, Error>
where
    Session: Clone,
    Validate: FnOnce(&mut Session) -> Result<Validated, Error>,
    Effect: for<'a> std::ops::AsyncFnOnce(
            &'a mut Session,
            Validated,
            &'a mut VisionStackOperationEffectBoundary,
        ) -> VisionStackOperationEffectResult<Value, Error>,
{
    let mut shadow = session.clone();
    let mut effect_boundary = VisionStackOperationEffectBoundary::new();
    let validation = validate(&mut shadow);
    let validated = match validation {
        Ok(validated) => validated,
        Err(error) => {
            let effect_result = effect_boundary.failure(error);
            return VisionStackOperationTransaction::from_effect_result(
                session,
                shadow,
                effect_result,
                effect_boundary,
            );
        }
    };
    let effect_result = effect(&mut shadow, validated, &mut effect_boundary).await;
    VisionStackOperationTransaction::from_effect_result(
        session,
        shadow,
        effect_result,
        effect_boundary,
    )
}

pub(crate) fn collect_vision_stack_session_resources<Key, Input, Resource, Error, Inputs, Build>(
    resources: Inputs,
    mut build: Build,
) -> Result<BTreeMap<Key, Resource>, Error>
where
    Key: Ord,
    Inputs: IntoIterator<Item = (Key, Input)>,
    Build: FnMut(Input) -> Result<Resource, Error>,
{
    resources
        .into_iter()
        .map(|(key, input)| build(input).map(|resource| (key, resource)))
        .collect()
}

pub(crate) enum VisionStackAsyncOperation {
    Start,
    Layer,
    Finish,
}

#[must_use]
pub(crate) const fn resolve_vision_stack_completion_action(
    operation: VisionStackAsyncOperation,
    outcome: Result<(), VisionStackGpuEffectBoundary>,
) -> CompletionAction {
    match outcome {
        Err(error) => match error {
            VisionStackGpuEffectBoundary::PreEffect => CompletionAction::Restore,
            VisionStackGpuEffectBoundary::PostEffect => CompletionAction::Finish,
        },
        Ok(()) => match operation {
            VisionStackAsyncOperation::Start | VisionStackAsyncOperation::Layer => {
                CompletionAction::Restore
            }
            VisionStackAsyncOperation::Finish => CompletionAction::Finish,
        },
    }
}

pub(crate) fn complete_vision_stack_async_operation<Session, T, E>(
    owner: &mut AsyncSessionOwner<Session>,
    lease: AsyncSessionLease,
    operation: VisionStackAsyncOperation,
    transaction: VisionStackOperationTransaction<Session, T, E>,
) -> (CompletionOutcome, Result<T, E>) {
    let VisionStackOperationTransaction {
        original,
        shadow,
        outcome,
        tracker_id,
        boundary,
    } = transaction;
    enforce_causal_invariant(
        tracker_id != 0,
        "vision-stack completion received an invalid effect tracker",
    );
    let boundary_result = match &outcome {
        Ok(_) => Ok(()),
        Err(failure) => {
            enforce_causal_invariant(
                failure.tracker_id == tracker_id && failure.boundary == boundary,
                "vision-stack completion received mismatched failure authority",
            );
            Err(failure.boundary)
        }
    };
    let action = resolve_vision_stack_completion_action(operation, boundary_result);
    let session = match (action, boundary) {
        (CompletionAction::Restore, VisionStackGpuEffectBoundary::PreEffect) => original,
        _ => shadow,
    };
    let returned = match outcome {
        Ok(value) => Ok(value),
        Err(failure) => Err(failure.error),
    };
    let completion = owner.complete(lease, session, action);
    (completion, returned)
}

#[must_use]
pub(crate) fn coordinate_vision_stack_completion_busy(
    execution_busy: &Cell<bool>,
    outcome: CompletionOutcome,
) -> CompletionOutcome {
    match outcome {
        CompletionOutcome::Finished | CompletionOutcome::Cancelled => execution_busy.set(false),
        CompletionOutcome::Restored | CompletionOutcome::Stale => {}
    }
    outcome
}
