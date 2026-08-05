use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use crate::{
    AsyncSessionOwner, CompletionAction, CompletionOutcome, SessionOwnerError,
    vision_stack_causal::{
        VISION_STACK_STREAMING_WEIGHT_SLOTS, VisionStackAsyncOperation,
        VisionStackErrorScopeAuthority, VisionStackErrorScopePopAttempt,
        VisionStackGpuEffectBoundary, VisionStackOperationEffectBoundary,
        VisionStackOperationEffectResult, VisionStackOperationTransaction,
        VisionStackPostEffectToken, VisionStackResidentCacheDisposition,
        VisionStackResidentWeightCache, VisionStackStreamingFailure,
        VisionStackStreamingLayerSchedule, VisionStackStreamingWeightCache,
        VisionStackStreamingWeightRange, collect_vision_stack_session_resources,
        complete_vision_stack_async_operation, coordinate_vision_stack_completion_busy,
        drain_vision_stack_error_scopes, observe_vision_stack_error_scope_pop,
        push_vision_stack_error_scope_or_drain,
        require_vision_stack_error_scope_admission_available,
        resolve_vision_stack_completion_action, run_vision_stack_error_scoped_operation,
        run_vision_stack_operation_transaction, run_vision_stack_resident_cold_layer,
        run_vision_stack_streaming_layer, run_vision_stack_streaming_session_layer,
    },
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct JournalSession {
    value: u32,
    journal: Rc<RefCell<Vec<&'static str>>>,
}

#[derive(Debug, Eq, PartialEq)]
struct JournalError(&'static str);

#[derive(Debug)]
struct PipelineResource {
    build: u32,
    drops: Rc<Cell<u32>>,
}

impl Drop for PipelineResource {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

fn poll_ready<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("causal unit fixture unexpectedly suspended"),
    }
}

fn owner_with_session(session: JournalSession) -> AsyncSessionOwner<JournalSession> {
    let owner = AsyncSessionOwner::new();
    owner.begin(session).unwrap();
    owner
}

#[test]
fn validation_failure_logs_no_marker_or_effect() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let original = JournalSession {
        value: 7,
        journal: Rc::clone(&journal),
    };
    let mut owner = owner_with_session(original.clone());
    let (lease, session) = owner.acquire().unwrap();
    let transaction: VisionStackOperationTransaction<JournalSession, (), JournalError> =
        poll_ready(run_vision_stack_operation_transaction(
            session,
            |shadow| {
                shadow.journal.borrow_mut().push("validate");
                shadow.value = 99;
                Err(JournalError("invalid"))
            },
            async |_shadow, _validated: (), _boundary| {
                panic!("validation failure invoked the effect closure")
            },
        ));
    let (completion, result) = complete_vision_stack_async_operation(
        &mut owner,
        lease,
        VisionStackAsyncOperation::Start,
        transaction,
    );

    assert_eq!(completion, CompletionOutcome::Restored);
    assert_eq!(result, Err(JournalError("invalid")));
    assert_eq!(owner.stored().unwrap().value, original.value);
    assert_eq!(journal.borrow().as_slice(), ["validate"]);
}

#[test]
fn first_effect_preparation_failure_is_restored_without_marker_or_raw_call() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let original = JournalSession {
        value: 41,
        journal: Rc::clone(&journal),
    };
    let mut owner = owner_with_session(original.clone());
    let (lease, session) = owner.acquire().unwrap();
    let transaction: VisionStackOperationTransaction<JournalSession, (), JournalError> =
        poll_ready(run_vision_stack_operation_transaction(
            session,
            |shadow| {
                shadow.journal.borrow_mut().push("validate");
                shadow.value = 42;
                shadow.journal.borrow_mut().push("prepare_first_effect");
                Err(JournalError("first-effect preparation"))
            },
            async move |_shadow, _prepared: (), _boundary| {
                panic!("failed first-effect preparation invoked the effect closure")
            },
        ));
    let (completion, result) = complete_vision_stack_async_operation(
        &mut owner,
        lease,
        VisionStackAsyncOperation::Start,
        transaction,
    );
    journal.borrow_mut().push("completion");

    assert_eq!(completion, CompletionOutcome::Restored);
    assert_eq!(result, Err(JournalError("first-effect preparation")));
    assert_eq!(owner.stored().unwrap().value, original.value);
    assert_eq!(
        journal.borrow().as_slice(),
        ["validate", "prepare_first_effect", "completion"],
    );
}

#[test]
fn failed_start_drops_partial_local_pipelines_and_new_begin_rebuilds_from_scratch() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let persistent_cache = Rc::new(RefCell::new(BTreeMap::from([("legacy", 17_u32)])));
    let persistent_snapshot = persistent_cache.borrow().clone();
    let builds = Rc::new(Cell::new(0_u32));
    let drops = Rc::new(Cell::new(0_u32));
    let initial = JournalSession {
        value: 0,
        journal: Rc::clone(&journal),
    };
    let mut owner = owner_with_session(initial.clone());

    let (lease, session) = owner.acquire().unwrap();
    let failed_builds = Rc::clone(&builds);
    let failed_drops = Rc::clone(&drops);
    let failed: VisionStackOperationTransaction<JournalSession, (), JournalError> =
        poll_ready(run_vision_stack_operation_transaction(
            session,
            |shadow| {
                shadow.journal.borrow_mut().push("validate");
                Ok(())
            },
            async move |shadow, (), boundary| {
                boundary
                    .run_webgpu_effect(async move |_post_effect: &VisionStackPostEffectToken| {
                        shadow.journal.borrow_mut().push("marker");
                        shadow.journal.borrow_mut().push("first_raw");
                        let session_local_pipelines = collect_vision_stack_session_resources(
                            [("pipeline-0", "pipeline-0"), ("pipeline-1", "pipeline-1")],
                            |label| {
                                let build = failed_builds.get() + 1;
                                failed_builds.set(build);
                                if label == "pipeline-1" {
                                    Err(JournalError("second pipeline rejected"))
                                } else {
                                    Ok(PipelineResource {
                                        build,
                                        drops: Rc::clone(&failed_drops),
                                    })
                                }
                            },
                        )?;
                        shadow.value = session_local_pipelines.len() as u32;
                        Ok::<(), JournalError>(())
                    })
                    .await
            },
        ));
    let (failed_completion, failed_result) = complete_vision_stack_async_operation(
        &mut owner,
        lease,
        VisionStackAsyncOperation::Start,
        failed,
    );
    journal.borrow_mut().push("completion");
    assert_eq!(failed_completion, CompletionOutcome::Finished);
    assert_eq!(failed_result, Err(JournalError("second pipeline rejected")));
    assert!(owner.stored().is_none());
    assert_eq!(
        drops.get(),
        1,
        "failed collection retained its first partial pipeline resource",
    );
    assert_eq!(&*persistent_cache.borrow(), &persistent_snapshot);

    owner.begin(initial).unwrap();
    let (lease, session) = owner.acquire().unwrap();
    let retry_builds = Rc::clone(&builds);
    let retry_drops = Rc::clone(&drops);
    let retried = poll_ready(run_vision_stack_operation_transaction(
        session,
        |shadow| {
            shadow.journal.borrow_mut().push("validate");
            Ok(())
        },
        async move |shadow, (), boundary| {
            boundary
                .run_webgpu_effect(async move |_post_effect: &VisionStackPostEffectToken| {
                    shadow.journal.borrow_mut().push("marker");
                    shadow.journal.borrow_mut().push("first_raw");
                    let session_local_pipelines = collect_vision_stack_session_resources(
                        [("pipeline-0", "pipeline-0"), ("pipeline-1", "pipeline-1")],
                        |_label| {
                            let build = retry_builds.get() + 1;
                            retry_builds.set(build);
                            Ok::<_, JournalError>(PipelineResource {
                                build,
                                drops: Rc::clone(&retry_drops),
                            })
                        },
                    )?;
                    let exact_builds = session_local_pipelines
                        .iter()
                        .map(|(label, resource)| (*label, resource.build))
                        .collect::<BTreeMap<_, _>>();
                    assert_eq!(
                        exact_builds,
                        BTreeMap::from([("pipeline-0", 3), ("pipeline-1", 4)]),
                        "retry collector changed a pipeline key or reused a partial resource",
                    );
                    shadow.value = session_local_pipelines.len() as u32;
                    Ok::<(), JournalError>(())
                })
                .await
        },
    ));
    let (retry_completion, retry_result) = complete_vision_stack_async_operation(
        &mut owner,
        lease,
        VisionStackAsyncOperation::Start,
        retried,
    );
    journal.borrow_mut().push("completion");

    assert_eq!(retry_completion, CompletionOutcome::Restored);
    assert_eq!(retry_result, Ok(()));
    assert_eq!(builds.get(), 4, "retry reused the failed partial pipeline");
    assert_eq!(owner.stored().unwrap().value, 2);
    assert_eq!(&*persistent_cache.borrow(), &persistent_snapshot);
    assert_eq!(
        journal.borrow().as_slice(),
        [
            "validate",
            "marker",
            "first_raw",
            "completion",
            "validate",
            "marker",
            "first_raw",
            "completion",
        ],
    );
}

#[test]
fn success_logs_validate_marker_first_raw_completion() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let initial = JournalSession {
        value: 1,
        journal: Rc::clone(&journal),
    };
    let mut owner = owner_with_session(initial);
    let (lease, session) = owner.acquire().unwrap();
    let transaction = poll_ready(run_vision_stack_operation_transaction(
        session,
        |shadow| {
            shadow.journal.borrow_mut().push("validate");
            shadow.value = 2;
            Ok(())
        },
        async move |shadow, (), boundary| {
            boundary
                .run_webgpu_effect(async move |post_effect: &VisionStackPostEffectToken| {
                    assert_ne!(post_effect.effect_tracker_id(), 0);
                    assert_eq!(
                        post_effect.effect_boundary(),
                        VisionStackGpuEffectBoundary::PostEffect,
                    );
                    shadow.journal.borrow_mut().push("marker");
                    shadow.journal.borrow_mut().push("first_raw");
                    shadow.value = 3;
                    Ok::<_, JournalError>(())
                })
                .await
        },
    ));
    let (completion, result) = complete_vision_stack_async_operation(
        &mut owner,
        lease,
        VisionStackAsyncOperation::Start,
        transaction,
    );
    journal.borrow_mut().push("completion");

    assert_eq!(completion, CompletionOutcome::Restored);
    assert_eq!(result, Ok(()));
    assert_eq!(owner.stored().unwrap().value, 3);
    assert_eq!(
        journal.borrow().as_slice(),
        ["validate", "marker", "first_raw", "completion"]
    );
}

#[test]
fn post_effect_failure_is_terminal() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let initial = JournalSession {
        value: 10,
        journal: Rc::clone(&journal),
    };
    let mut owner = owner_with_session(initial);
    let generation = owner.generation();
    let (lease, session) = owner.acquire().unwrap();
    let transaction = poll_ready(run_vision_stack_operation_transaction(
        session,
        |shadow| {
            shadow.journal.borrow_mut().push("validate");
            shadow.value = 11;
            Ok(())
        },
        async move |shadow, (), boundary| {
            boundary
                .run_webgpu_effect(async move |post_effect: &VisionStackPostEffectToken| {
                    assert_ne!(post_effect.effect_tracker_id(), 0);
                    assert_eq!(
                        post_effect.effect_boundary(),
                        VisionStackGpuEffectBoundary::PostEffect
                    );
                    shadow.journal.borrow_mut().push("marker");
                    shadow.journal.borrow_mut().push("first_raw");
                    shadow.value = 12;
                    Err::<(), _>(JournalError("post"))
                })
                .await
        },
    ));
    let (completion, result) = complete_vision_stack_async_operation(
        &mut owner,
        lease,
        VisionStackAsyncOperation::Start,
        transaction,
    );
    journal.borrow_mut().push("completion");

    assert_eq!(completion, CompletionOutcome::Finished);
    assert_eq!(result, Err(JournalError("post")));
    assert_eq!(owner.generation(), None);
    assert!(owner.stored().is_none());
    assert_ne!(generation, owner.generation());
    assert_eq!(
        journal.borrow().as_slice(),
        ["validate", "marker", "first_raw", "completion"]
    );
}

#[test]
fn duplicate_effect_run_is_rejected() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let mut boundary = VisionStackOperationEffectBoundary::new();
    let first_journal = Rc::clone(&journal);
    let first: VisionStackOperationEffectResult<(), JournalError> = poll_ready(
        boundary.run_webgpu_effect(async move |post_effect: &VisionStackPostEffectToken| {
            assert_eq!(
                post_effect.effect_boundary(),
                VisionStackGpuEffectBoundary::PostEffect
            );
            first_journal.borrow_mut().push("marker");
            first_journal.borrow_mut().push("first_raw");
            Ok(())
        }),
    );
    assert_eq!(
        first.effect_boundary(),
        VisionStackGpuEffectBoundary::PostEffect
    );

    let second_invoked = Rc::new(RefCell::new(false));
    let second_probe = Rc::clone(&second_invoked);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _: VisionStackOperationEffectResult<(), JournalError> = poll_ready(
            boundary.run_webgpu_effect(async move |_post_effect: &VisionStackPostEffectToken| {
                *second_probe.borrow_mut() = true;
                Ok(())
            }),
        );
    }));
    assert!(panic.is_err());
    assert!(!*second_invoked.borrow());
    assert_eq!(journal.borrow().as_slice(), ["marker", "first_raw"]);
}

#[test]
fn stale_pre_effect_result_is_rejected() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let session = JournalSession {
        value: 20,
        journal: Rc::clone(&journal),
    };
    let shadow = session.clone();
    let mut boundary = VisionStackOperationEffectBoundary::new();
    let stale: VisionStackOperationEffectResult<(), JournalError> =
        boundary.failure(JournalError("stale-pre"));
    let effect_journal = Rc::clone(&journal);
    let _: VisionStackOperationEffectResult<(), JournalError> = poll_ready(
        boundary.run_webgpu_effect(async move |post_effect: &VisionStackPostEffectToken| {
            assert_eq!(
                post_effect.effect_boundary(),
                VisionStackGpuEffectBoundary::PostEffect
            );
            effect_journal.borrow_mut().push("marker");
            effect_journal.borrow_mut().push("first_raw");
            Ok(())
        }),
    );
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ =
            VisionStackOperationTransaction::from_effect_result(session, shadow, stale, boundary);
    }));
    assert!(panic.is_err());
    assert_eq!(journal.borrow().as_slice(), ["marker", "first_raw"]);
}

#[test]
fn completion_policy_covers_every_operation_and_boundary() {
    for (case, operation, outcome, expected) in [
        (
            "start success",
            VisionStackAsyncOperation::Start,
            Ok(()),
            CompletionAction::Restore,
        ),
        (
            "layer success",
            VisionStackAsyncOperation::Layer,
            Ok(()),
            CompletionAction::Restore,
        ),
        (
            "finish success",
            VisionStackAsyncOperation::Finish,
            Ok(()),
            CompletionAction::Finish,
        ),
        (
            "start pre-effect failure",
            VisionStackAsyncOperation::Start,
            Err(VisionStackGpuEffectBoundary::PreEffect),
            CompletionAction::Restore,
        ),
        (
            "layer pre-effect failure",
            VisionStackAsyncOperation::Layer,
            Err(VisionStackGpuEffectBoundary::PreEffect),
            CompletionAction::Restore,
        ),
        (
            "finish pre-effect failure",
            VisionStackAsyncOperation::Finish,
            Err(VisionStackGpuEffectBoundary::PreEffect),
            CompletionAction::Restore,
        ),
        (
            "start post-effect failure",
            VisionStackAsyncOperation::Start,
            Err(VisionStackGpuEffectBoundary::PostEffect),
            CompletionAction::Finish,
        ),
        (
            "layer post-effect failure",
            VisionStackAsyncOperation::Layer,
            Err(VisionStackGpuEffectBoundary::PostEffect),
            CompletionAction::Finish,
        ),
        (
            "finish post-effect failure",
            VisionStackAsyncOperation::Finish,
            Err(VisionStackGpuEffectBoundary::PostEffect),
            CompletionAction::Finish,
        ),
    ] {
        assert_eq!(
            resolve_vision_stack_completion_action(operation, outcome),
            expected,
            "completion policy mismatch for {case}",
        );
    }
}

#[test]
fn busy_coordinator_clears_only_terminal_outcomes() {
    for (outcome, initially_busy, expected_busy) in [
        (CompletionOutcome::Restored, true, true),
        (CompletionOutcome::Restored, false, false),
        (CompletionOutcome::Stale, true, true),
        (CompletionOutcome::Stale, false, false),
        (CompletionOutcome::Finished, true, false),
        (CompletionOutcome::Finished, false, false),
        (CompletionOutcome::Cancelled, true, false),
        (CompletionOutcome::Cancelled, false, false),
    ] {
        let execution_busy = Cell::new(initially_busy);

        assert_eq!(
            coordinate_vision_stack_completion_busy(&execution_busy, outcome),
            outcome,
        );
        assert_eq!(
            execution_busy.get(),
            expected_busy,
            "outcome: {outcome:?}, initially_busy: {initially_busy}",
        );
    }
}

#[test]
fn cross_tracker_effect_result_is_rejected() {
    let journal = Rc::new(RefCell::new(Vec::new()));
    let session = JournalSession {
        value: 30,
        journal: Rc::clone(&journal),
    };
    let shadow = session.clone();
    let first_boundary = VisionStackOperationEffectBoundary::new();
    let effect_result: VisionStackOperationEffectResult<(), JournalError> =
        first_boundary.failure(JournalError("foreign-pre"));
    let second_boundary = VisionStackOperationEffectBoundary::new();

    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = VisionStackOperationTransaction::from_effect_result(
            session,
            shadow,
            effect_result,
            second_boundary,
        );
    }));

    assert!(panic.is_err());
    assert!(journal.borrow().is_empty());
}

fn scope_authority<'a>(
    healthy: &'a Cell<bool>,
    occupied: &'a Cell<bool>,
) -> VisionStackErrorScopeAuthority<'a, &'static str> {
    VisionStackErrorScopeAuthority::acquire(
        healthy,
        occupied,
        JournalError("scope authority poisoned"),
        JournalError("scope authority occupied"),
    )
    .expect("fresh scope authority must be admitted")
}

fn push_scope(
    authority: &mut VisionStackErrorScopeAuthority<'_, &'static str>,
    scope: &'static str,
) {
    poll_ready(push_vision_stack_error_scope_or_drain(
        authority,
        scope,
        |_scope| Ok::<(), JournalError>(()),
        |scope| async move {
            VisionStackErrorScopePopAttempt::<&str, JournalError>::Popped(Ok(scope))
        },
    ))
    .expect("scripted scope push must succeed");
}

#[test]
fn every_scope_push_failure_drains_exact_prior_lifo_and_releases_admission() {
    for failed_scope_index in 0..3 {
        let scopes = ["internal", "out_of_memory", "validation"];
        let healthy = Cell::new(true);
        let occupied = Cell::new(false);
        let journal = Rc::new(RefCell::new(Vec::new()));
        let mut authority = scope_authority(&healthy, &occupied);

        let mut observed_failure = None;
        for (index, scope) in scopes.into_iter().enumerate() {
            let push_journal = Rc::clone(&journal);
            let pop_journal = Rc::clone(&journal);
            let result = poll_ready(push_vision_stack_error_scope_or_drain(
                &mut authority,
                scope,
                move |scope| {
                    push_journal.borrow_mut().push(match *scope {
                        "internal" => "push:internal",
                        "out_of_memory" => "push:out_of_memory",
                        "validation" => "push:validation",
                        _ => panic!("unexpected scope"),
                    });
                    if index == failed_scope_index {
                        Err(JournalError("push rejected"))
                    } else {
                        Ok(())
                    }
                },
                move |scope| {
                    let pop_journal = Rc::clone(&pop_journal);
                    async move {
                        pop_journal.borrow_mut().push(scope);
                        VisionStackErrorScopePopAttempt::Popped(Ok(scope))
                    }
                },
            ));
            if index == failed_scope_index {
                observed_failure = Some(result.expect_err("selected push must fail"));
                break;
            }
            assert!(result.is_ok());
        }

        let (push_error, cleanup) = observed_failure
            .expect("every scenario must exercise one push failure")
            .into_parts();
        let (popped, cleanup_errors, remaining) = cleanup.into_parts();
        assert_eq!(push_error, JournalError("push rejected"));
        let expected_pops = scopes[..failed_scope_index]
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(popped, expected_pops);
        assert!(cleanup_errors.is_empty());
        assert_eq!(remaining, 0);
        assert!(authority.is_empty());
        assert!(healthy.get());
        assert!(occupied.get());
        let expected_journal = scopes[..=failed_scope_index]
            .iter()
            .map(|scope| match *scope {
                "internal" => "push:internal",
                "out_of_memory" => "push:out_of_memory",
                "validation" => "push:validation",
                _ => unreachable!(),
            })
            .chain(expected_pops.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(*journal.borrow(), expected_journal);

        drop(authority);
        assert!(!occupied.get());
        assert!(healthy.get());
        let mut retry = scope_authority(&healthy, &occupied);
        for scope in scopes {
            push_scope(&mut retry, scope);
        }
        let retry_cleanup = poll_ready(drain_vision_stack_error_scopes(
            &mut retry,
            |scope| async move {
                VisionStackErrorScopePopAttempt::<&str, JournalError>::Popped(Ok(scope))
            },
        ));
        let (retry_popped, retry_errors, retry_remaining) = retry_cleanup.into_parts();
        assert_eq!(retry_popped, ["validation", "out_of_memory", "internal"]);
        assert!(retry_errors.is_empty());
        assert_eq!(retry_remaining, 0);
        drop(retry);
        assert!(!occupied.get());
        assert!(healthy.get());
    }
}

#[test]
fn confirmed_pop_normalization_failures_are_recorded_while_lower_scopes_are_drained() {
    let lifo = ["validation", "out_of_memory", "internal"];
    for fail_at in 0..lifo.len() {
        let healthy = Cell::new(true);
        let occupied = Cell::new(false);
        let mut authority = scope_authority(&healthy, &occupied);
        for scope in ["internal", "out_of_memory", "validation"] {
            push_scope(&mut authority, scope);
        }

        let attempts = Rc::new(RefCell::new(Vec::new()));
        let attempt_probe = Rc::clone(&attempts);
        let drained = poll_ready(drain_vision_stack_error_scopes(
            &mut authority,
            move |scope| {
                let index = attempt_probe.borrow().len();
                attempt_probe.borrow_mut().push(scope);
                async move {
                    if index == fail_at {
                        VisionStackErrorScopePopAttempt::Popped(Err(JournalError(
                            "normalization failed",
                        )))
                    } else {
                        VisionStackErrorScopePopAttempt::Popped(Ok(scope))
                    }
                }
            },
        ));
        let (popped, failures, remaining) = drained.into_parts();
        assert_eq!(*attempts.borrow(), lifo);
        assert_eq!(
            popped,
            lifo.iter()
                .enumerate()
                .filter_map(|(index, scope)| (index != fail_at).then_some(*scope))
                .collect::<Vec<_>>(),
        );
        assert_eq!(failures, [JournalError("normalization failed")]);
        assert_eq!(remaining, 0);
        assert!(authority.is_empty());
        assert!(healthy.get());
        drop(authority);
        assert!(!occupied.get());
    }
}

#[test]
fn every_unconfirmed_pop_retains_exact_top_poisons_and_blocks_raw_reentry() {
    let lifo = ["validation", "out_of_memory", "internal"];
    for fail_at in 0_u32..3 {
        let healthy = Cell::new(true);
        let occupied = Cell::new(false);
        let mut authority = scope_authority(&healthy, &occupied);
        for scope in ["internal", "out_of_memory", "validation"] {
            push_scope(&mut authority, scope);
        }

        let attempts = Rc::new(RefCell::new(Vec::new()));
        let attempt_probe = Rc::clone(&attempts);
        let index = Rc::new(Cell::new(0_u32));
        let index_probe = Rc::clone(&index);
        let first = poll_ready(drain_vision_stack_error_scopes(
            &mut authority,
            move |scope| {
                let current = index_probe.get();
                index_probe.set(current + 1);
                attempt_probe.borrow_mut().push(scope);
                async move {
                    if current == fail_at {
                        VisionStackErrorScopePopAttempt::NotPopped(JournalError("pop unconfirmed"))
                    } else {
                        VisionStackErrorScopePopAttempt::Popped(Ok(scope))
                    }
                }
            },
        ));
        let (popped, failures, remaining) = first.into_parts();
        assert_eq!(popped, lifo[..fail_at as usize]);
        assert_eq!(failures, [JournalError("pop unconfirmed")]);
        assert_eq!(remaining, 3 - fail_at as usize);
        assert_eq!(*attempts.borrow(), lifo[..=fail_at as usize]);
        assert_eq!(authority.scopes(), &lifo[fail_at as usize..]);
        assert!(!healthy.get());
        assert!(occupied.get());

        let blocked_attempts = Rc::new(Cell::new(0_u32));
        let blocked_probe = Rc::clone(&blocked_attempts);
        let blocked = poll_ready(drain_vision_stack_error_scopes(
            &mut authority,
            move |scope| {
                blocked_probe.set(blocked_probe.get() + 1);
                async move { VisionStackErrorScopePopAttempt::<&str, JournalError>::Popped(Ok(scope)) }
            },
        ));
        let (blocked_popped, blocked_failures, blocked_remaining) = blocked.into_parts();
        assert!(blocked_popped.is_empty());
        assert!(blocked_failures.is_empty());
        assert_eq!(blocked_remaining, remaining);
        assert_eq!(blocked_attempts.get(), 0);
        assert_eq!(authority.scopes(), &lifo[fail_at as usize..]);

        drop(authority);
        assert!(!occupied.get());
        assert!(matches!(
            VisionStackErrorScopeAuthority::<&str>::acquire(
                &healthy,
                &occupied,
                JournalError("scope authority poisoned"),
                JournalError("scope authority occupied"),
            ),
            Err(JournalError("scope authority poisoned"))
        ));
        assert!(!occupied.get());

        let replacement_healthy = Cell::new(true);
        let replacement_occupied = Cell::new(false);
        let replacement = scope_authority(&replacement_healthy, &replacement_occupied);
        drop(replacement);
        assert!(replacement_healthy.get());
        assert!(!replacement_occupied.get());
    }
}

#[test]
fn device_scope_admission_is_exclusive_and_empty_drop_is_recoverable() {
    let poisoned_health = Cell::new(false);
    let unoccupied = Cell::new(false);
    assert_eq!(
        require_vision_stack_error_scope_admission_available(
            &poisoned_health,
            &unoccupied,
            JournalError("scope authority poisoned"),
            JournalError("scope authority occupied"),
        ),
        Err(JournalError("scope authority poisoned")),
    );
    assert!(!unoccupied.get());

    let healthy = Cell::new(true);
    let occupied = Cell::new(false);
    let authority = scope_authority(&healthy, &occupied);
    assert!(occupied.get());
    assert_eq!(
        require_vision_stack_error_scope_admission_available(
            &healthy,
            &occupied,
            JournalError("scope authority poisoned"),
            JournalError("scope authority occupied"),
        ),
        Err(JournalError("scope authority occupied")),
    );
    assert!(matches!(
        VisionStackErrorScopeAuthority::<&str>::acquire(
            &healthy,
            &occupied,
            JournalError("scope authority poisoned"),
            JournalError("scope authority occupied"),
        ),
        Err(JournalError("scope authority occupied"))
    ));
    drop(authority);
    assert!(healthy.get());
    assert!(!occupied.get());
    let retry = scope_authority(&healthy, &occupied);
    drop(retry);
    assert!(healthy.get());
    assert!(!occupied.get());
}

#[test]
fn operation_error_after_first_push_is_returned_only_after_full_lifo_drain() {
    let healthy = Cell::new(true);
    let occupied = Cell::new(false);
    let journal = Rc::new(RefCell::new(Vec::new()));
    let mut authority = scope_authority(&healthy, &occupied);
    for scope in ["internal", "out_of_memory", "validation"] {
        push_scope(&mut authority, scope);
    }
    let pop_journal = Rc::clone(&journal);
    let completed = poll_ready(run_vision_stack_error_scoped_operation(
        authority,
        async { Err::<(), _>(JournalError("post-push finish branch failed")) },
        move |scope| {
            let pop_journal = Rc::clone(&pop_journal);
            async move {
                pop_journal.borrow_mut().push(scope);
                VisionStackErrorScopePopAttempt::Popped(Ok(scope))
            }
        },
    ));
    let (operation, cleanup) = completed.into_parts();
    let (popped, failures, remaining) = cleanup.into_parts();
    assert_eq!(
        operation,
        Err(JournalError("post-push finish branch failed"))
    );
    assert_eq!(popped, ["validation", "out_of_memory", "internal"]);
    assert!(failures.is_empty());
    assert_eq!(remaining, 0);
    assert_eq!(
        *journal.borrow(),
        ["validation", "out_of_memory", "internal"]
    );
    assert!(healthy.get());
    assert!(!occupied.get());
}

#[test]
fn dropping_partial_push_operation_or_pop_future_poisons_and_releases_admission() {
    {
        let healthy = Cell::new(true);
        let occupied = Cell::new(false);
        let mut future = Box::pin(async {
            let mut authority = scope_authority(&healthy, &occupied);
            push_scope(&mut authority, "internal");
            let _ = push_vision_stack_error_scope_or_drain(
                &mut authority,
                "out_of_memory",
                |_scope| Err::<(), _>(JournalError("push rejected")),
                |_scope| {
                    std::future::pending::<VisionStackErrorScopePopAttempt<&str, JournalError>>()
                },
            )
            .await;
        });
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert!(healthy.get());
        assert!(occupied.get());
        drop(future);
        assert!(!healthy.get());
        assert!(!occupied.get());
    }

    {
        let healthy = Cell::new(true);
        let occupied = Cell::new(false);
        let mut future = Box::pin(async {
            let mut authority = scope_authority(&healthy, &occupied);
            for scope in ["internal", "out_of_memory", "validation"] {
                push_scope(&mut authority, scope);
            }
            let _ = run_vision_stack_error_scoped_operation(
                authority,
                std::future::pending::<Result<(), JournalError>>(),
                |scope| async move {
                    VisionStackErrorScopePopAttempt::<&str, JournalError>::Popped(Ok(scope))
                },
            )
            .await;
        });
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert!(healthy.get());
        assert!(occupied.get());
        drop(future);
        assert!(!healthy.get());
        assert!(!occupied.get());
    }

    {
        let healthy = Cell::new(true);
        let occupied = Cell::new(false);
        let mut future = Box::pin(async {
            let mut authority = scope_authority(&healthy, &occupied);
            for scope in ["internal", "out_of_memory", "validation"] {
                push_scope(&mut authority, scope);
            }
            let _ = run_vision_stack_error_scoped_operation(
                authority,
                async { Ok::<_, JournalError>(()) },
                |_scope| {
                    std::future::pending::<VisionStackErrorScopePopAttempt<&str, JournalError>>()
                },
            )
            .await;
        });
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert!(healthy.get());
        assert!(occupied.get());
        drop(future);
        assert!(!healthy.get());
        assert!(!occupied.get());
    }
}

#[test]
fn two_stage_pop_adapter_distinguishes_unconfirmed_rejection_from_post_pop_failure() {
    let call_error = poll_ready(observe_vision_stack_error_scope_pop(
        Err::<&str, _>(JournalError("call0 failed")),
        |_pending| async { Ok::<&str, JournalError>("resolved") },
        Ok::<_, JournalError>,
    ));
    assert!(matches!(
        call_error,
        VisionStackErrorScopePopAttempt::NotPopped(JournalError("call0 failed"))
    ));

    let promise_rejection = poll_ready(observe_vision_stack_error_scope_pop(
        Ok::<_, JournalError>("promise"),
        |_pending| async { Err::<&str, _>(JournalError("promise rejected")) },
        Ok::<_, JournalError>,
    ));
    assert!(matches!(
        promise_rejection,
        VisionStackErrorScopePopAttempt::NotPopped(JournalError("promise rejected"))
    ));

    let normalization_error = poll_ready(observe_vision_stack_error_scope_pop(
        Ok::<_, JournalError>("promise"),
        |_pending| async { Ok::<_, JournalError>("resolved") },
        |_value| Err::<&str, _>(JournalError("normalization failed")),
    ));
    assert!(matches!(
        normalization_error,
        VisionStackErrorScopePopAttempt::Popped(Err(JournalError("normalization failed")))
    ));

    let confirmed = poll_ready(observe_vision_stack_error_scope_pop(
        Ok::<_, JournalError>("promise"),
        |_pending| async { Ok::<_, JournalError>("resolved") },
        Ok::<_, JournalError>,
    ));
    assert!(matches!(
        confirmed,
        VisionStackErrorScopePopAttempt::Popped(Ok("resolved"))
    ));
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StreamingResource {
    id: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StreamingSession {
    next_layer: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum StreamingEvent {
    Validate {
        layer: usize,
    },
    Allocate {
        slot: usize,
        bytes: u64,
        resource: usize,
    },
    Upload {
        layer: usize,
        slot: usize,
        offset: u64,
        bytes: u64,
        resource: usize,
    },
    Submit {
        layer: usize,
        layer_index: u32,
        checkpoint_slot: Option<usize>,
        resources: Vec<usize>,
    },
}

const OFFICIAL_VISION_HIDDEN_SIZE: u64 = 1_152;
const OFFICIAL_VISION_INTERMEDIATE_SIZE: u64 = 4_304;

fn official_streaming_weight_ranges() -> [VisionStackStreamingWeightRange; 16] {
    let hidden_vector = OFFICIAL_VISION_HIDDEN_SIZE * 4;
    let hidden_matrix = OFFICIAL_VISION_HIDDEN_SIZE * OFFICIAL_VISION_HIDDEN_SIZE * 4;
    let fc1_matrix = OFFICIAL_VISION_INTERMEDIATE_SIZE * OFFICIAL_VISION_HIDDEN_SIZE * 4;
    let intermediate_vector = OFFICIAL_VISION_INTERMEDIATE_SIZE * 4;
    let fc2_matrix = OFFICIAL_VISION_HIDDEN_SIZE * OFFICIAL_VISION_INTERMEDIATE_SIZE * 4;
    let sizes = [
        hidden_vector,
        hidden_vector,
        hidden_matrix,
        hidden_vector,
        hidden_matrix,
        hidden_vector,
        hidden_matrix,
        hidden_vector,
        hidden_matrix,
        hidden_vector,
        hidden_vector,
        hidden_vector,
        fc1_matrix,
        intermediate_vector,
        fc2_matrix,
        hidden_vector,
    ];
    let mut offset = 0;
    sizes.map(|length| {
        let range = VisionStackStreamingWeightRange::new(offset, length);
        offset += length;
        range
    })
}

fn official_checkpoint_slot(layer: usize) -> Option<usize> {
    match layer {
        0 => Some(0),
        1 => Some(1),
        13 => Some(2),
        26 => Some(3),
        _ => None,
    }
}

fn streaming_schedule(layer: usize) -> VisionStackStreamingLayerSchedule {
    VisionStackStreamingLayerSchedule::new(
        u32::try_from(layer).unwrap(),
        official_checkpoint_slot(layer),
        official_streaming_weight_ranges(),
    )
}

#[test]
fn resident_weight_cache_turns_the_second_27_layer_run_into_zero_upload_reuse() {
    let key = "paddleocr-vl-1.6/fp16-input-major".to_owned();
    let mut cache = VisionStackResidentWeightCache::new();
    let events = RefCell::new(Vec::new());
    let mut session = StreamingSession { next_layer: 0 };

    assert_eq!(
        cache.prepare(key.clone(), 27).unwrap(),
        VisionStackResidentCacheDisposition::Cold,
    );
    for layer in 0..27 {
        run_vision_stack_resident_cold_layer(
            &mut cache,
            &mut session,
            |shadow| {
                assert_eq!(shadow.next_layer, layer);
                shadow.next_layer += 1;
                events.borrow_mut().push(StreamingEvent::Validate { layer });
                Ok::<_, JournalError>(streaming_schedule(layer))
            },
            |slot, range| {
                let resource = StreamingResource {
                    id: layer * VISION_STACK_STREAMING_WEIGHT_SLOTS + slot,
                };
                events.borrow_mut().push(StreamingEvent::Allocate {
                    slot,
                    bytes: range.length_bytes(),
                    resource: resource.id,
                });
                Ok(resource)
            },
            |slot, range, resource| {
                events.borrow_mut().push(StreamingEvent::Upload {
                    layer,
                    slot,
                    offset: range.offset_bytes(),
                    bytes: range.length_bytes(),
                    resource: resource.id,
                });
                Ok(())
            },
            |shadow, layer_index, checkpoint_slot, resources| {
                assert_eq!(shadow.next_layer, layer + 1);
                events.borrow_mut().push(StreamingEvent::Submit {
                    layer,
                    layer_index,
                    checkpoint_slot,
                    resources: resources.iter().map(|resource| resource.id).collect(),
                });
                Ok(())
            },
        )
        .unwrap();
    }
    assert!(
        !cache.is_ready_for(&key, 27),
        "27 resident layers without authenticated post-norm weights must remain cold",
    );
    cache
        .store_post_norm(vec![
            StreamingResource { id: 10_000 },
            StreamingResource { id: 10_001 },
        ])
        .unwrap();

    assert_eq!(session.next_layer, 27);
    let cold_events = events.into_inner();
    assert_eq!(
        cold_events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Validate { .. }))
            .count(),
        27,
    );
    assert_eq!(
        cold_events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Allocate { .. }))
            .count(),
        27 * VISION_STACK_STREAMING_WEIGHT_SLOTS,
    );
    assert_eq!(
        cold_events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Upload { .. }))
            .count(),
        27 * VISION_STACK_STREAMING_WEIGHT_SLOTS,
    );
    assert_eq!(
        cold_events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Submit { .. }))
            .count(),
        27,
    );
    assert!(cache.is_ready_for(&key, 27));
    assert_eq!(
        cache.prepare(key.clone(), 27).unwrap(),
        VisionStackResidentCacheDisposition::Ready,
    );

    let mut observed = Vec::new();
    for layer in 0..27 {
        let resources = cache.clone_layer(layer).unwrap();
        observed.extend(resources.into_iter().map(|resource| resource.id));
    }
    let post_norm = cache
        .clone_post_norm()
        .unwrap()
        .into_iter()
        .map(|resource| resource.id)
        .collect::<Vec<_>>();

    assert_eq!(observed.len(), 27 * VISION_STACK_STREAMING_WEIGHT_SLOTS);
    assert_eq!(
        observed,
        (0..27 * VISION_STACK_STREAMING_WEIGHT_SLOTS).collect::<Vec<_>>(),
    );
    assert_eq!(post_norm, [10_000, 10_001]);
}

#[test]
fn resident_cold_path_commits_no_cache_entry_before_authenticated_submit_succeeds() {
    for failure in ["validation", "upload", "submit"] {
        let mut cache = VisionStackResidentWeightCache::new();
        cache.prepare("model".to_owned(), 27).unwrap();
        let mut session = StreamingSession { next_layer: 0 };
        let effects = Cell::new(0);

        let result = run_vision_stack_resident_cold_layer(
            &mut cache,
            &mut session,
            |shadow| {
                if failure == "validation" {
                    return Err(JournalError("validation"));
                }
                shadow.next_layer += 1;
                Ok(streaming_schedule(0))
            },
            |slot, _range| {
                effects.set(effects.get() + 1);
                Ok(StreamingResource { id: slot })
            },
            |_slot, _range, _resource| {
                effects.set(effects.get() + 1);
                if failure == "upload" {
                    Err(JournalError("upload"))
                } else {
                    Ok(())
                }
            },
            |_shadow, _layer_index, _checkpoint_slot, _resources| {
                effects.set(effects.get() + 1);
                if failure == "submit" {
                    Err(JournalError("submit"))
                } else {
                    Ok(())
                }
            },
        );

        assert!(result.is_err(), "{failure} unexpectedly succeeded");
        assert_eq!(
            cache.loaded_layer_count(),
            0,
            "{failure} leaked unauthenticated or unsubmitted resources into the resident cache",
        );
        assert!(!cache.is_ready_for(&"model".to_owned(), 27));
        assert_eq!(
            session.next_layer, 0,
            "{failure} committed the shadow protocol session",
        );
        if failure == "validation" {
            assert_eq!(effects.get(), 0, "validation failure leaked a GPU effect");
        }
    }
}

#[test]
fn resident_weight_cache_never_reuses_partial_or_different_model_weights() {
    let mut cache = VisionStackResidentWeightCache::new();
    let first_key = "model-a/fp16-input-major".to_owned();
    let second_key = "model-b/fp16-input-major".to_owned();

    assert_eq!(
        cache.prepare(first_key.clone(), 27).unwrap(),
        VisionStackResidentCacheDisposition::Cold,
    );
    cache
        .store_layer(
            0,
            (0..VISION_STACK_STREAMING_WEIGHT_SLOTS)
                .map(|id| StreamingResource { id })
                .collect(),
        )
        .unwrap();
    assert!(!cache.is_ready_for(&first_key, 27));
    assert!(cache.clone_layer(0).is_err());

    assert_eq!(
        cache.prepare(first_key.clone(), 27).unwrap(),
        VisionStackResidentCacheDisposition::Cold,
        "an interrupted first load must restart instead of becoming a cache hit",
    );
    assert_eq!(cache.loaded_layer_count(), 0);

    for layer in 0..27 {
        cache
            .store_layer(
                layer,
                (0..VISION_STACK_STREAMING_WEIGHT_SLOTS)
                    .map(|slot| StreamingResource {
                        id: layer * VISION_STACK_STREAMING_WEIGHT_SLOTS + slot,
                    })
                    .collect(),
            )
            .unwrap();
    }
    cache
        .store_post_norm(vec![
            StreamingResource { id: 10_000 },
            StreamingResource { id: 10_001 },
        ])
        .unwrap();
    assert!(cache.is_ready_for(&first_key, 27));

    assert_eq!(
        cache.prepare(second_key.clone(), 27).unwrap(),
        VisionStackResidentCacheDisposition::Cold,
    );
    assert!(!cache.is_ready_for(&first_key, 27));
    assert!(!cache.is_ready_for(&second_key, 27));
    assert_eq!(cache.loaded_layer_count(), 0);
    assert!(cache.clone_layer(0).is_err());
    assert!(cache.clone_post_norm().is_err());
}

#[test]
fn resident_weight_cache_rejects_wrong_order_and_resource_cardinality_without_mutation() {
    let mut cache = VisionStackResidentWeightCache::new();
    cache.prepare("model".to_owned(), 27).unwrap();

    let wrong_order = cache.store_layer(
        1,
        (0..VISION_STACK_STREAMING_WEIGHT_SLOTS)
            .map(|id| StreamingResource { id })
            .collect(),
    );
    assert!(wrong_order.is_err());
    assert_eq!(cache.loaded_layer_count(), 0);

    let short_layer = cache.store_layer(
        0,
        (0..VISION_STACK_STREAMING_WEIGHT_SLOTS - 1)
            .map(|id| StreamingResource { id })
            .collect(),
    );
    assert!(short_layer.is_err());
    assert_eq!(cache.loaded_layer_count(), 0);

    let early_post_norm = cache.store_post_norm(vec![
        StreamingResource { id: 10_000 },
        StreamingResource { id: 10_001 },
    ]);
    assert!(early_post_norm.is_err());
    assert!(!cache.is_ready_for(&"model".to_owned(), 27));
}

#[test]
fn streaming_authority_runs_27_layers_in_exact_order_with_one_reused_weight_set() {
    assert_eq!(VISION_STACK_STREAMING_WEIGHT_SLOTS, 16);
    let events = RefCell::new(Vec::new());
    let mut cache = VisionStackStreamingWeightCache::new();
    let mut session = StreamingSession { next_layer: 0 };
    let ranges = official_streaming_weight_ranges();

    for layer in 0..27 {
        let outcome = run_vision_stack_streaming_layer(
            &mut cache,
            &mut session,
            |shadow| {
                assert_eq!(shadow.next_layer, layer);
                shadow.next_layer += 1;
                events.borrow_mut().push(StreamingEvent::Validate { layer });
                Ok::<_, JournalError>(streaming_schedule(layer))
            },
            |slot, range| {
                let resource = StreamingResource { id: slot + 1_000 };
                events.borrow_mut().push(StreamingEvent::Allocate {
                    slot,
                    bytes: range.length_bytes(),
                    resource: resource.id,
                });
                Ok(resource)
            },
            |slot, range, resource| {
                events.borrow_mut().push(StreamingEvent::Upload {
                    layer,
                    slot,
                    offset: range.offset_bytes(),
                    bytes: range.length_bytes(),
                    resource: resource.id,
                });
                Ok(())
            },
            |shadow, layer_index, checkpoint_slot, resources| {
                assert_eq!(shadow.next_layer, layer + 1);
                events.borrow_mut().push(StreamingEvent::Submit {
                    layer,
                    layer_index,
                    checkpoint_slot,
                    resources: resources.iter().map(|resource| resource.id).collect(),
                });
                Ok(())
            },
        );
        assert_eq!(outcome, Ok(()), "streaming layer {layer} failed");
    }
    assert_eq!(session.next_layer, 27);

    let events = events.into_inner();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Validate { .. }))
            .count(),
        27,
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Allocate { .. }))
            .count(),
        VISION_STACK_STREAMING_WEIGHT_SLOTS,
        "the 27-layer path allocated more than one bounded weight set",
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Upload { .. }))
            .count(),
        27 * VISION_STACK_STREAMING_WEIGHT_SLOTS,
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamingEvent::Submit { .. }))
            .count(),
        27,
    );

    let resource_ids = (0..VISION_STACK_STREAMING_WEIGHT_SLOTS)
        .map(|slot| slot + 1_000)
        .collect::<Vec<_>>();
    let mut cursor = 0;
    for layer in 0..27 {
        assert_eq!(events[cursor], StreamingEvent::Validate { layer });
        cursor += 1;
        if layer == 0 {
            for slot in 0..VISION_STACK_STREAMING_WEIGHT_SLOTS {
                assert_eq!(
                    events[cursor],
                    StreamingEvent::Allocate {
                        slot,
                        bytes: ranges[slot].length_bytes(),
                        resource: resource_ids[slot],
                    },
                );
                cursor += 1;
            }
        }
        for slot in 0..VISION_STACK_STREAMING_WEIGHT_SLOTS {
            assert_eq!(
                events[cursor],
                StreamingEvent::Upload {
                    layer,
                    slot,
                    offset: ranges[slot].offset_bytes(),
                    bytes: ranges[slot].length_bytes(),
                    resource: resource_ids[slot],
                },
            );
            cursor += 1;
        }
        assert_eq!(
            events[cursor],
            StreamingEvent::Submit {
                layer,
                layer_index: u32::try_from(layer).unwrap(),
                checkpoint_slot: official_checkpoint_slot(layer),
                resources: resource_ids.clone(),
            },
        );
        cursor += 1;
    }
    assert_eq!(cursor, events.len());
    assert_eq!(
        cache
            .resources()
            .unwrap()
            .iter()
            .map(|resource| resource.id)
            .collect::<Vec<_>>(),
        resource_ids,
    );
    assert_eq!(
        ranges
            .iter()
            .map(VisionStackStreamingWeightRange::length_bytes)
            .sum::<u64>(),
        60_958_016,
        "fixture drifted from the official 1152x4304 layer layout",
    );
}

#[test]
fn streaming_admission_and_cache_shape_failures_have_zero_gpu_effects() {
    for error in [
        "wrong digest",
        "wrong layer order",
        "non-finite tensor",
        "no stored session",
    ] {
        let effects = Cell::new(0);
        let mut cache = VisionStackStreamingWeightCache::<StreamingResource>::new();
        let mut session = StreamingSession { next_layer: 0 };
        let outcome = run_vision_stack_streaming_layer(
            &mut cache,
            &mut session,
            |shadow| {
                shadow.next_layer = 99;
                Err::<VisionStackStreamingLayerSchedule, _>(JournalError(error))
            },
            |_slot, _range| {
                effects.set(effects.get() + 1);
                Ok(StreamingResource { id: 1 })
            },
            |_slot, _range, _resource| {
                effects.set(effects.get() + 1);
                Ok(())
            },
            |_shadow, _layer_index, _checkpoint_slot, _resources| {
                effects.set(effects.get() + 1);
                Ok(())
            },
        );

        assert_eq!(
            outcome,
            Err(VisionStackStreamingFailure::Admission(JournalError(error))),
        );
        assert_eq!(effects.get(), 0, "{error} leaked a GPU effect");
        assert!(cache.resources().is_none());
        assert_eq!(
            session.next_layer, 0,
            "{error} committed rejected protocol state",
        );
    }

    let effects = Cell::new(0);
    let mut cache = VisionStackStreamingWeightCache::new();
    let mut session = StreamingSession { next_layer: 0 };
    assert_eq!(
        run_vision_stack_streaming_layer(
            &mut cache,
            &mut session,
            |shadow| {
                shadow.next_layer += 1;
                Ok::<_, JournalError>(streaming_schedule(0))
            },
            |slot, _range| Ok(StreamingResource { id: slot }),
            |_slot, _range, _resource| Ok(()),
            |_shadow, _layer_index, _checkpoint_slot, _resources| Ok(()),
        ),
        Ok(()),
    );
    assert_eq!(session.next_layer, 1);
    let cached_ids = cache
        .resources()
        .unwrap()
        .iter()
        .map(|resource| resource.id)
        .collect::<Vec<_>>();

    let admission = run_vision_stack_streaming_layer(
        &mut cache,
        &mut session,
        |shadow| {
            shadow.next_layer = 99;
            Err::<VisionStackStreamingLayerSchedule, _>(JournalError("wrong digest"))
        },
        |_slot, _range| {
            effects.set(effects.get() + 1);
            Ok(StreamingResource { id: 99 })
        },
        |_slot, _range, _resource| {
            effects.set(effects.get() + 1);
            Ok(())
        },
        |_shadow, _layer_index, _checkpoint_slot, _resources| {
            effects.set(effects.get() + 1);
            Ok(())
        },
    );
    assert_eq!(
        admission,
        Err(VisionStackStreamingFailure::Admission(JournalError(
            "wrong digest",
        ))),
    );
    assert_eq!(session.next_layer, 1);
    assert_eq!(
        cache
            .resources()
            .unwrap()
            .iter()
            .map(|resource| resource.id)
            .collect::<Vec<_>>(),
        cached_ids,
    );

    let ranges = official_streaming_weight_ranges();
    let mismatched = VisionStackStreamingLayerSchedule::new(
        1,
        None,
        std::array::from_fn(|slot| {
            VisionStackStreamingWeightRange::new(
                ranges[slot].offset_bytes(),
                if slot == 7 {
                    9_999
                } else {
                    ranges[slot].length_bytes()
                },
            )
        }),
    );
    let outcome = run_vision_stack_streaming_layer(
        &mut cache,
        &mut session,
        |shadow| {
            shadow.next_layer += 1;
            Ok::<_, JournalError>(mismatched)
        },
        |_slot, _range| {
            effects.set(effects.get() + 1);
            Ok(StreamingResource { id: 99 })
        },
        |_slot, _range, _resource| {
            effects.set(effects.get() + 1);
            Ok(())
        },
        |_shadow, _layer_index, _checkpoint_slot, _resources| {
            effects.set(effects.get() + 1);
            Ok(())
        },
    );
    assert!(matches!(
        outcome,
        Err(VisionStackStreamingFailure::CacheLengthMismatch {
            slot: 7,
            expected_bytes: 4_608,
            actual_bytes: 9_999,
        })
    ));
    assert_eq!(effects.get(), 0);
    assert_eq!(
        session.next_layer, 1,
        "cache mismatch committed accepted protocol state",
    );
    assert_eq!(
        cache
            .resources()
            .unwrap()
            .iter()
            .map(|resource| resource.id)
            .collect::<Vec<_>>(),
        cached_ids,
        "a rejected layer changed the reusable weight set",
    );
}

#[test]
fn streaming_stops_at_the_first_effect_failure_and_reports_post_effect() {
    for (
        case,
        warm_cache,
        fail_allocate,
        fail_upload,
        fail_submit,
        expected_error,
        expected_effects,
    ) in [
        (
            "allocate",
            false,
            Some(3),
            None,
            false,
            "allocation failed",
            4,
        ),
        ("upload", true, None, Some(5), false, "upload failed", 6),
        (
            "submit",
            true,
            None,
            None,
            true,
            "submit failed",
            VISION_STACK_STREAMING_WEIGHT_SLOTS + 1,
        ),
    ] {
        let effects = RefCell::new(Vec::new());
        let mut cache = VisionStackStreamingWeightCache::new();
        let mut session = StreamingSession { next_layer: 0 };
        if warm_cache {
            assert_eq!(
                run_vision_stack_streaming_layer(
                    &mut cache,
                    &mut session,
                    |shadow| {
                        shadow.next_layer += 1;
                        Ok::<_, JournalError>(streaming_schedule(0))
                    },
                    |slot, _range| Ok(StreamingResource { id: slot }),
                    |_slot, _range, _resource| Ok(()),
                    |_shadow, _layer_index, _checkpoint_slot, _resources| Ok(()),
                ),
                Ok(()),
            );
        }
        let layer = session.next_layer;
        let outcome = run_vision_stack_streaming_layer(
            &mut cache,
            &mut session,
            |shadow| {
                shadow.next_layer += 1;
                Ok(streaming_schedule(layer))
            },
            |slot, _range| {
                effects.borrow_mut().push(("allocate", slot));
                if fail_allocate == Some(slot) {
                    Err(JournalError("allocation failed"))
                } else {
                    Ok(StreamingResource { id: slot })
                }
            },
            |slot, _range, _resource| {
                effects.borrow_mut().push(("upload", slot));
                if fail_upload == Some(slot) {
                    Err(JournalError("upload failed"))
                } else {
                    Ok(())
                }
            },
            |_shadow, _layer_index, _checkpoint_slot, _resources| {
                effects.borrow_mut().push(("submit", 0));
                if fail_submit {
                    Err(JournalError("submit failed"))
                } else {
                    Ok(())
                }
            },
        );

        match outcome {
            Err(VisionStackStreamingFailure::Effect { error, boundary }) => {
                assert_eq!(error, JournalError(expected_error), "{case}");
                assert_eq!(boundary, VisionStackGpuEffectBoundary::PostEffect, "{case}",);
            }
            other => panic!("{case} failure was not terminal/post-effect: {other:?}"),
        }
        assert_eq!(
            effects.borrow().len(),
            expected_effects,
            "{case} failure allowed a later effect",
        );
        assert!(
            !effects.borrow().iter().any(|(kind, _)| *kind == "submit") || fail_submit,
            "{case} failure unexpectedly submitted the layer",
        );
        if case == "allocate" {
            assert!(
                cache.resources().is_none(),
                "partial allocation escaped into the persistent cache",
            );
        }
    }
}

#[test]
fn streaming_session_coordinator_commits_rolls_back_terminates_and_reuses_cache() {
    let owner = AsyncSessionOwner::new();
    owner.begin(StreamingSession { next_layer: 0 }).unwrap();
    let generation = owner.generation();
    let execution_busy = Cell::new(true);
    let mut cache = VisionStackStreamingWeightCache::new();

    assert_eq!(
        run_vision_stack_streaming_session_layer(
            &owner,
            &execution_busy,
            &mut cache,
            |shadow| {
                shadow.next_layer += 1;
                Ok::<_, JournalError>(streaming_schedule(0))
            },
            |slot, _range| Ok(StreamingResource { id: 1_000 + slot }),
            |_slot, _range, _resource| Ok(()),
            |_shadow, _layer_index, _checkpoint_slot, _resources| Ok(()),
        ),
        Ok(()),
    );
    assert_eq!(owner.generation(), generation);
    assert_eq!(owner.stored().unwrap().next_layer, 1);
    assert!(execution_busy.get());
    let cached_ids = cache
        .resources()
        .unwrap()
        .iter()
        .map(|resource| resource.id)
        .collect::<Vec<_>>();

    assert_eq!(
        run_vision_stack_streaming_session_layer(
            &owner,
            &execution_busy,
            &mut cache,
            |shadow| {
                shadow.next_layer = 99;
                Err::<VisionStackStreamingLayerSchedule, _>(JournalError("wrong digest"))
            },
            |_slot, _range| panic!("admission failure allocated"),
            |_slot, _range, _resource| panic!("admission failure uploaded"),
            |_shadow, _layer_index, _checkpoint_slot, _resources| {
                panic!("admission failure submitted")
            },
        ),
        Err(VisionStackStreamingFailure::Admission(JournalError(
            "wrong digest",
        ))),
    );
    assert_eq!(owner.generation(), generation);
    assert_eq!(owner.stored().unwrap().next_layer, 1);
    assert!(execution_busy.get());
    assert_eq!(
        cache
            .resources()
            .unwrap()
            .iter()
            .map(|resource| resource.id)
            .collect::<Vec<_>>(),
        cached_ids,
    );

    let mut incompatible_ranges = official_streaming_weight_ranges();
    incompatible_ranges[0] =
        VisionStackStreamingWeightRange::new(incompatible_ranges[0].offset_bytes(), 9_999);
    assert_eq!(
        run_vision_stack_streaming_session_layer(
            &owner,
            &execution_busy,
            &mut cache,
            |shadow| {
                shadow.next_layer += 1;
                Ok::<_, JournalError>(VisionStackStreamingLayerSchedule::new(
                    1,
                    None,
                    incompatible_ranges,
                ))
            },
            |_slot, _range| panic!("incompatible geometry allocated"),
            |_slot, _range, _resource| panic!("incompatible geometry uploaded"),
            |_shadow, _layer_index, _checkpoint_slot, _resources| {
                panic!("incompatible geometry submitted")
            },
        ),
        Err(VisionStackStreamingFailure::CacheLengthMismatch {
            slot: 0,
            expected_bytes: 4_608,
            actual_bytes: 9_999,
        }),
        "incompatible second geometry must be a retryable pre-effect mismatch",
    );
    assert_eq!(owner.generation(), generation);
    assert_eq!(owner.stored().unwrap().next_layer, 1);
    assert!(execution_busy.get());
    assert_eq!(
        cache
            .resources()
            .unwrap()
            .iter()
            .map(|resource| resource.id)
            .collect::<Vec<_>>(),
        cached_ids,
    );

    let terminal = run_vision_stack_streaming_session_layer(
        &owner,
        &execution_busy,
        &mut cache,
        |shadow| {
            shadow.next_layer += 1;
            Ok(streaming_schedule(1))
        },
        |_slot, _range| panic!("warm cache allocated"),
        |_slot, _range, _resource| Ok(()),
        |_shadow, _layer_index, _checkpoint_slot, _resources| Err(JournalError("submit failed")),
    );
    assert_eq!(
        terminal,
        Err(VisionStackStreamingFailure::Effect {
            error: JournalError("submit failed"),
            boundary: VisionStackGpuEffectBoundary::PostEffect,
        }),
    );
    assert!(owner.stored().is_none());
    assert_eq!(owner.generation(), None);
    assert!(!execution_busy.get());

    owner.begin(StreamingSession { next_layer: 0 }).unwrap();
    execution_busy.set(true);
    let allocations = Cell::new(0);
    assert_eq!(
        run_vision_stack_streaming_session_layer(
            &owner,
            &execution_busy,
            &mut cache,
            |shadow| {
                shadow.next_layer += 1;
                Ok::<_, JournalError>(streaming_schedule(0))
            },
            |_slot, _range| {
                allocations.set(allocations.get() + 1);
                Ok(StreamingResource { id: 9_999 })
            },
            |_slot, _range, _resource| Ok(()),
            |_shadow, _layer_index, _checkpoint_slot, _resources| Ok(()),
        ),
        Ok(()),
    );
    assert_eq!(
        allocations.get(),
        0,
        "a compatible second OCR session rebuilt the bounded cache",
    );
    assert_eq!(owner.stored().unwrap().next_layer, 1);
    assert!(execution_busy.get());

    let unavailable_owner = AsyncSessionOwner::<StreamingSession>::new();
    let unavailable_busy = Cell::new(false);
    let unavailable = run_vision_stack_streaming_session_layer(
        &unavailable_owner,
        &unavailable_busy,
        &mut cache,
        |_shadow| -> Result<VisionStackStreamingLayerSchedule, JournalError> {
            panic!("unavailable session validated")
        },
        |_slot, _range| panic!("unavailable session allocated"),
        |_slot, _range, _resource| panic!("unavailable session uploaded"),
        |_shadow, _layer_index, _checkpoint_slot, _resources| {
            panic!("unavailable session submitted")
        },
    );
    assert_eq!(
        unavailable,
        Err(VisionStackStreamingFailure::Unavailable(
            SessionOwnerError::NoStoredSession,
        )),
    );
    assert!(!unavailable_busy.get());
}
