//! Host-side contract for generation-aware ownership across async boundaries.
//!
//! `WebRuntime` can store its real browser session in this platform-neutral
//! owner. Acquiring moves that session to the future; completing decides
//! whether the same session may be restored after an `await`.

use std::{cell::RefCell, rc::Rc};

use pvlc_runtime_web::{
    AbortDisposition, AsyncSessionLease, AsyncSessionOwner, AsyncSessionOwnerError,
    CompletionAction, CompletionOutcome,
};

#[derive(Debug)]
struct DropProbe {
    identity: &'static str,
    preflight_observations: usize,
    drops: Rc<RefCell<Vec<&'static str>>>,
}

impl DropProbe {
    fn new(identity: &'static str, drops: &Rc<RefCell<Vec<&'static str>>>) -> Self {
        Self {
            identity,
            preflight_observations: 0,
            drops: Rc::clone(drops),
        }
    }
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.drops.borrow_mut().push(self.identity);
    }
}

fn assert_lease_generation(lease: &AsyncSessionLease, expected: u64) {
    assert_eq!(lease.generation(), expected);
}

fn duplicate_callback_token(lease: &AsyncSessionLease) -> AsyncSessionLease {
    lease.clone()
}

fn dropped(drops: &Rc<RefCell<Vec<&'static str>>>) -> Vec<&'static str> {
    drops.borrow().clone()
}

#[test]
fn deferred_abort_keeps_the_old_session_owned_until_completion_drops_it() {
    let drops = Rc::new(RefCell::new(Vec::new()));
    let owner = AsyncSessionOwner::new();
    let generation = owner
        .begin(DropProbe::new("generation-one", &drops))
        .expect("first session should begin");
    assert!(owner.is_busy());
    assert_eq!(
        owner.stored().map(|session| session.identity),
        Some("generation-one")
    );

    let (lease, session) = owner
        .acquire()
        .expect("stored session should move to its async operation");
    assert_lease_generation(&lease, generation);
    assert_eq!(session.identity, "generation-one");
    assert!(owner.is_busy());
    assert!(owner.is_in_flight());
    assert!(owner.stored().is_none());

    assert_eq!(owner.abort(), AbortDisposition::Deferred);
    assert_eq!(owner.abort(), AbortDisposition::Deferred);
    assert!(
        owner.is_busy(),
        "abort must not release an owned async session"
    );
    assert_eq!(
        owner.begin(DropProbe::new("forbidden", &drops)),
        Err(AsyncSessionOwnerError::Busy),
    );
    assert_eq!(dropped(&drops), ["forbidden"]);

    assert_eq!(
        owner.complete(lease, session, CompletionAction::Restore),
        CompletionOutcome::Cancelled,
        "an aborted operation must drop rather than restore its session",
    );
    assert_eq!(dropped(&drops), ["forbidden", "generation-one"]);
    assert!(!owner.is_busy());
    assert!(!owner.is_in_flight());
    assert!(owner.stored().is_none());

    let next_generation = owner
        .begin(DropProbe::new("generation-two", &drops))
        .expect("cancel completion should release the owner");
    assert!(next_generation > generation);
    assert_eq!(
        owner.stored().map(|session| session.identity),
        Some("generation-two")
    );
}

#[test]
fn abort_of_a_stored_session_drops_it_immediately_and_is_idempotent_while_idle() {
    let drops = Rc::new(RefCell::new(Vec::new()));
    let owner = AsyncSessionOwner::new();
    let generation = owner
        .begin(DropProbe::new("stored", &drops))
        .expect("session should begin");
    assert_eq!(owner.generation(), Some(generation));

    assert_eq!(owner.abort(), AbortDisposition::Released);
    assert_eq!(dropped(&drops), ["stored"]);
    assert!(!owner.is_busy());
    assert!(owner.stored().is_none());
    assert!(!owner.is_in_flight());

    assert_eq!(owner.abort(), AbortDisposition::AlreadyIdle);
    assert_eq!(dropped(&drops), ["stored"]);
    let next_generation = owner
        .begin(DropProbe::new("next", &drops))
        .expect("immediate abort should permit a new session");
    assert!(next_generation > generation);
}

#[test]
fn stale_callback_while_newer_generation_is_in_flight_drops_only_the_stale_payload() {
    let drops = Rc::new(RefCell::new(Vec::new()));
    let owner = AsyncSessionOwner::new();
    let old_generation = owner
        .begin(DropProbe::new("old", &drops))
        .expect("old session should begin");
    let (old_lease, old_session) = owner.acquire().expect("old session should be acquirable");
    let duplicate_callback_token = duplicate_callback_token(&old_lease);
    assert_eq!(
        owner.complete(old_lease, old_session, CompletionAction::Finish),
        CompletionOutcome::Finished,
    );
    assert_eq!(dropped(&drops), ["old"]);

    let new_generation = owner
        .begin(DropProbe::new("new", &drops))
        .expect("new session should begin");
    assert!(new_generation > old_generation);
    let (new_lease, new_session) = owner.acquire().expect("new session should be in flight");
    assert_lease_generation(&new_lease, new_generation);

    assert_eq!(
        owner.complete(
            duplicate_callback_token,
            DropProbe::new("stale-callback", &drops),
            CompletionAction::Restore,
        ),
        CompletionOutcome::Stale,
    );
    assert_eq!(dropped(&drops), ["old", "stale-callback"]);
    assert_eq!(owner.generation(), Some(new_generation));
    assert!(owner.is_busy());
    assert!(owner.is_in_flight());
    assert!(owner.stored().is_none());
    assert_eq!(
        new_session.identity, "new",
        "newer payload must remain untouched"
    );

    assert_eq!(
        owner.complete(new_lease, new_session, CompletionAction::Restore),
        CompletionOutcome::Restored,
    );
    assert_eq!(owner.generation(), Some(new_generation));
    assert_eq!(owner.stored().map(|session| session.identity), Some("new"));
    assert_eq!(dropped(&drops), ["old", "stale-callback"]);
}

#[test]
fn synchronous_stored_access_normal_restore_and_terminal_finish_preserve_identity() {
    let drops = Rc::new(RefCell::new(Vec::new()));
    let owner = AsyncSessionOwner::new();
    assert!(matches!(
        owner.acquire(),
        Err(AsyncSessionOwnerError::NoStoredSession)
    ));

    let generation = owner
        .begin(DropProbe::new("normal", &drops))
        .expect("session should begin");
    owner
        .stored_mut()
        .expect("preflight needs synchronous stored-session access")
        .preflight_observations += 1;

    let (first_lease, first_session) = owner.acquire().expect("session should be acquirable");
    assert_eq!(first_session.identity, "normal");
    assert_eq!(first_session.preflight_observations, 1);
    assert_eq!(
        owner.complete(first_lease, first_session, CompletionAction::Restore),
        CompletionOutcome::Restored,
    );
    assert_eq!(owner.generation(), Some(generation));
    assert_eq!(
        owner
            .stored()
            .map(|session| (session.identity, session.preflight_observations)),
        Some(("normal", 1))
    );
    assert!(dropped(&drops).is_empty());

    let (terminal_lease, terminal_session) = owner
        .acquire()
        .expect("restored session should be acquirable again");
    assert_eq!(
        owner.complete(terminal_lease, terminal_session, CompletionAction::Finish),
        CompletionOutcome::Finished,
    );
    assert_eq!(dropped(&drops), ["normal"]);
    assert!(!owner.is_busy());
    assert!(owner.stored().is_none());
    assert!(!owner.is_in_flight());
    assert_eq!(owner.generation(), None);
}
