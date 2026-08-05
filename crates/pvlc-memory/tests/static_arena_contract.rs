use blake3::hash;
use proptest::prelude::*;

use pvlc_memory::{
    ArenaAllocation, ArenaConfig, ArenaErrorCode, StaticArenaPlan, TensorLifetime,
    plan_static_arena, verify_static_arena_plan,
};

const DEFAULT_BASE_ALIGNMENT: u64 = 64;
const DEFAULT_ARENA_ALIGNMENT: u64 = 64;
const HIDDEN_BYTES: u64 = 1_024 * 4;
const INTERMEDIATE_BYTES: u64 = 3_072 * 4;
const VISION_NO_ALIAS_BYTES: u64 = 11 * HIDDEN_BYTES + 2 * INTERMEDIATE_BYTES;
const VISION_ALIAS_LOWER_BOUND_BYTES: u64 = 28_672;
const SMALL_PLAN_BYTES: &str = concat!(
    "arena_bytes=256\n",
    "alpha|0|96|64|0|0|stage_a\n",
    "beta|0|64|128|1|2|stage_b\n",
    "gamma|0|80|64|3|3|stage_c\n",
);
const SMALL_PLAN_BLAKE3: &str = "68575448f3e9dad58acac3372eb9978a5a27e74d61ee6f958deb77a1a27951bc";

fn config(allow_aliasing: bool) -> ArenaConfig {
    ArenaConfig {
        allow_aliasing,
        arena_alignment: DEFAULT_ARENA_ALIGNMENT,
        base_alignment: DEFAULT_BASE_ALIGNMENT,
    }
}

fn aligned_config(allow_aliasing: bool, base_alignment: u64, arena_alignment: u64) -> ArenaConfig {
    ArenaConfig {
        allow_aliasing,
        base_alignment,
        arena_alignment,
    }
}

fn lifetime(
    id: &str,
    byte_size: u64,
    alignment: u64,
    first_write: u32,
    last_use: u32,
) -> TensorLifetime {
    TensorLifetime {
        id: id.to_owned(),
        byte_size,
        alignment,
        first_write,
        last_use,
        stage_label: None,
    }
}

fn labeled_lifetime(
    id: &str,
    stage_label: &str,
    byte_size: u64,
    alignment: u64,
    first_write: u32,
    last_use: u32,
) -> TensorLifetime {
    TensorLifetime {
        id: id.to_owned(),
        byte_size,
        alignment,
        first_write,
        last_use,
        stage_label: Some(stage_label.to_owned()),
    }
}

fn manual_small_expected_plan() -> StaticArenaPlan {
    StaticArenaPlan {
        arena_bytes: 256,
        allocations: vec![
            ArenaAllocation {
                id: "alpha".to_owned(),
                offset: 0,
                size: 96,
                alignment: 64,
                first_write: 0,
                last_use: 0,
                stage_label: Some("stage_a".to_owned()),
            },
            ArenaAllocation {
                id: "beta".to_owned(),
                offset: 0,
                size: 64,
                alignment: 128,
                first_write: 1,
                last_use: 2,
                stage_label: Some("stage_b".to_owned()),
            },
            ArenaAllocation {
                id: "gamma".to_owned(),
                offset: 0,
                size: 80,
                alignment: 64,
                first_write: 3,
                last_use: 3,
                stage_label: Some("stage_c".to_owned()),
            },
        ],
    }
}

fn small_plan_lifetimes() -> Vec<TensorLifetime> {
    vec![
        labeled_lifetime("beta", "stage_b", 64, 128, 1, 2),
        labeled_lifetime("gamma", "stage_c", 80, 32, 3, 3),
        labeled_lifetime("alpha", "stage_a", 96, 16, 0, 0),
    ]
}

fn canonical_plan_bytes(plan: &StaticArenaPlan) -> Vec<u8> {
    let mut bytes = format!("arena_bytes={}\n", plan.arena_bytes).into_bytes();
    for allocation in &plan.allocations {
        bytes.extend_from_slice(
            format!(
                "{}|{}|{}|{}|{}|{}|{}\n",
                allocation.id,
                allocation.offset,
                allocation.size,
                allocation.alignment,
                allocation.first_write,
                allocation.last_use,
                allocation.stage_label.as_deref().unwrap_or("<none>")
            )
            .as_bytes(),
        );
    }
    bytes
}

fn plan_digest_hex(plan: &StaticArenaPlan) -> String {
    hash(&canonical_plan_bytes(plan)).to_hex().to_string()
}

fn end_offset(allocation: &ArenaAllocation) -> u64 {
    allocation
        .offset
        .checked_add(allocation.size)
        .expect("allocation end overflowed during test inspection")
}

fn overlaps(left: &ArenaAllocation, right: &ArenaAllocation) -> bool {
    left.offset < end_offset(right) && right.offset < end_offset(left)
}

fn is_live_at(lifetime: &TensorLifetime, point: u32) -> bool {
    lifetime.first_write <= point && point <= lifetime.last_use
}

fn source_lifetime<'a>(lifetimes: &'a [TensorLifetime], id: &str) -> &'a TensorLifetime {
    lifetimes
        .iter()
        .find(|lifetime| lifetime.id == id)
        .expect("allocation missing source lifetime")
}

macro_rules! assert_error_code {
    ($result:expr, $expected:expr $(,)?) => {{
        let error = ($result).expect_err("expected planner/verifier error");
        assert_eq!(error.code(), $expected);
    }};
}

fn assert_aligned_bounded_and_disjoint(
    lifetimes: &[TensorLifetime],
    plan: &StaticArenaPlan,
    config: ArenaConfig,
) {
    assert_eq!(plan.allocations.len(), lifetimes.len());
    assert_eq!(plan.arena_bytes % config.arena_alignment, 0);

    for allocation in &plan.allocations {
        let lifetime = source_lifetime(lifetimes, &allocation.id);
        let effective_alignment = lifetime.alignment.max(config.base_alignment);
        assert_eq!(allocation.alignment, effective_alignment);
        assert_eq!(allocation.offset % effective_alignment, 0);
        assert!(end_offset(allocation) <= plan.arena_bytes);
    }

    let max_schedule_point = lifetimes
        .iter()
        .map(|lifetime| lifetime.last_use)
        .max()
        .unwrap_or(0);
    for point in 0..=max_schedule_point {
        let live = plan
            .allocations
            .iter()
            .filter(|allocation| is_live_at(source_lifetime(lifetimes, &allocation.id), point))
            .collect::<Vec<_>>();
        for (index, left) in live.iter().enumerate() {
            for right in &live[index + 1..] {
                assert!(
                    !overlaps(left, right),
                    "schedule point {point}: {} overlaps {}",
                    left.id,
                    right.id
                );
            }
        }
    }
}

fn collect_ids(plan: &StaticArenaPlan) -> Vec<&str> {
    plan.allocations
        .iter()
        .map(|allocation| allocation.id.as_str())
        .collect()
}

fn vision_layer_schedule() -> Vec<TensorLifetime> {
    vec![
        labeled_lifetime("dispatch00.current_input", "input", HIDDEN_BYTES, 64, 0, 6),
        labeled_lifetime("dispatch01.norm1", "norm1", HIDDEN_BYTES, 64, 0, 3),
        labeled_lifetime("dispatch02.query", "query", HIDDEN_BYTES, 64, 1, 4),
        labeled_lifetime("dispatch03.key", "key", HIDDEN_BYTES, 64, 2, 4),
        labeled_lifetime("dispatch04.value", "value", HIDDEN_BYTES, 64, 3, 4),
        labeled_lifetime(
            "dispatch05.attention_context",
            "attention-context",
            HIDDEN_BYTES,
            64,
            4,
            5,
        ),
        labeled_lifetime(
            "dispatch06.attention_output",
            "attention-output",
            HIDDEN_BYTES,
            64,
            5,
            6,
        ),
        labeled_lifetime(
            "dispatch07.attention_residual",
            "attention-residual",
            HIDDEN_BYTES,
            64,
            6,
            11,
        ),
        labeled_lifetime("dispatch08.norm2", "norm2", HIDDEN_BYTES, 64, 7, 8),
        labeled_lifetime(
            "dispatch09.mlp_fc1",
            "mlp-fc1",
            INTERMEDIATE_BYTES,
            64,
            8,
            9,
        ),
        labeled_lifetime(
            "dispatch10.mlp_activation",
            "mlp-activation",
            INTERMEDIATE_BYTES,
            64,
            9,
            10,
        ),
        labeled_lifetime(
            "dispatch11.mlp_output",
            "mlp-output",
            HIDDEN_BYTES,
            64,
            10,
            11,
        ),
        labeled_lifetime("dispatch12.output", "output", HIDDEN_BYTES, 64, 11, 12),
    ]
}

#[test]
fn empty_input_freezes_zero_arena_zero_allocations_and_verifies() {
    let plan = plan_static_arena(&[], config(true)).unwrap();
    assert_eq!(plan.arena_bytes, 0);
    assert!(plan.allocations.is_empty());
    verify_static_arena_plan(&[], &plan, config(true)).unwrap();
}

#[test]
fn no_alias_plan_uses_disjoint_aligned_ranges() {
    let lifetimes = vec![
        lifetime("input", 256, 64, 0, 3),
        lifetime("tmp_q", 128, 64, 1, 2),
        lifetime("output", 256, 64, 3, 4),
    ];
    let plan = plan_static_arena(&lifetimes, config(false)).unwrap();
    verify_static_arena_plan(&lifetimes, &plan, config(false)).unwrap();
    assert_aligned_bounded_and_disjoint(&lifetimes, &plan, config(false));
    assert_eq!(collect_ids(&plan), vec!["input", "tmp_q", "output"]);
}

#[test]
fn aliasing_requires_strictly_non_overlapping_inclusive_lifetimes() {
    let touching = vec![
        lifetime("left", 256, 64, 0, 1),
        lifetime("right", 256, 64, 1, 2),
    ];
    let touching_plan = plan_static_arena(&touching, config(true)).unwrap();
    verify_static_arena_plan(&touching, &touching_plan, config(true)).unwrap();
    assert_ne!(
        touching_plan.allocations[0].offset,
        touching_plan.allocations[1].offset
    );

    let strictly_disjoint = vec![
        lifetime("left", 256, 64, 0, 0),
        lifetime("right", 256, 64, 1, 2),
    ];
    let alias_plan = plan_static_arena(&strictly_disjoint, config(true)).unwrap();
    verify_static_arena_plan(&strictly_disjoint, &alias_plan, config(true)).unwrap();
    assert_eq!(
        alias_plan.allocations[0].offset,
        alias_plan.allocations[1].offset
    );
}

#[test]
fn base_alignment_lifts_tensor_alignment_and_arena_bytes_round_to_arena_alignment() {
    let cfg = aligned_config(false, 128, 256);
    let lifetimes = vec![
        lifetime("alpha", 96, 16, 0, 1),
        lifetime("beta", 64, 64, 0, 1),
    ];
    let plan = plan_static_arena(&lifetimes, cfg).unwrap();
    verify_static_arena_plan(&lifetimes, &plan, cfg).unwrap();
    assert_eq!(plan.allocations[0].alignment, 128);
    assert_eq!(plan.allocations[1].alignment, 128);
    assert_eq!(plan.allocations[0].offset, 0);
    assert_eq!(plan.allocations[1].offset, 128);
    assert_eq!(plan.arena_bytes, 256);
    assert_aligned_bounded_and_disjoint(&lifetimes, &plan, cfg);
}

#[test]
fn mixed_alignment_and_size_requirements_are_respected() {
    let cfg = aligned_config(false, 64, 256);
    let lifetimes = vec![
        lifetime("small16", 48, 16, 0, 1),
        lifetime("wide128", 512, 128, 2, 3),
        lifetime("mid64", 192, 64, 4, 5),
    ];
    let plan = plan_static_arena(&lifetimes, cfg).unwrap();
    verify_static_arena_plan(&lifetimes, &plan, cfg).unwrap();
    assert_eq!(plan.allocations[0].offset % 64, 0);
    assert_eq!(plan.allocations[1].offset % 128, 0);
    assert_eq!(plan.allocations[2].offset % 64, 0);
    assert_eq!(plan.arena_bytes % 256, 0);
    assert_aligned_bounded_and_disjoint(&lifetimes, &plan, cfg);
}

#[test]
fn permutation_is_deterministic_and_canonical_by_first_write_then_id() {
    let ordered = vec![
        lifetime("alpha", 64, 64, 0, 0),
        lifetime("beta", 64, 64, 0, 0),
        lifetime("gamma", 64, 64, 1, 1),
    ];
    let permuted = vec![ordered[2].clone(), ordered[1].clone(), ordered[0].clone()];

    let ordered_plan = plan_static_arena(&ordered, config(true)).unwrap();
    let permuted_plan = plan_static_arena(&permuted, config(true)).unwrap();
    verify_static_arena_plan(&ordered, &ordered_plan, config(true)).unwrap();
    verify_static_arena_plan(&permuted, &permuted_plan, config(true)).unwrap();

    assert_eq!(collect_ids(&ordered_plan), vec!["alpha", "beta", "gamma"]);
    assert_eq!(collect_ids(&ordered_plan), collect_ids(&permuted_plan));
    assert_eq!(
        canonical_plan_bytes(&ordered_plan),
        canonical_plan_bytes(&permuted_plan)
    );
}

#[test]
fn canonical_small_plan_freezes_exact_bytes_and_blake3_snapshot() {
    let cfg = aligned_config(true, 64, 256);
    let lifetimes = small_plan_lifetimes();
    let expected = manual_small_expected_plan();
    let plan = plan_static_arena(&lifetimes, cfg).unwrap();
    verify_static_arena_plan(&lifetimes, &plan, cfg).unwrap();

    assert_eq!(collect_ids(&plan), vec!["alpha", "beta", "gamma"]);
    assert_eq!(canonical_plan_bytes(&expected), SMALL_PLAN_BYTES.as_bytes());
    assert_eq!(plan_digest_hex(&expected), SMALL_PLAN_BLAKE3);
    assert_eq!(canonical_plan_bytes(&plan), SMALL_PLAN_BYTES.as_bytes());
}

#[test]
fn rejects_duplicate_ids_and_invalid_lifetimes_with_exact_error_codes() {
    assert_error_code!(
        plan_static_arena(&[lifetime("", 64, 64, 0, 0)], config(false)),
        ArenaErrorCode::EmptyId,
    );
    assert_error_code!(
        plan_static_arena(
            &[
                lifetime("dup", 128, 64, 0, 0),
                lifetime("dup", 256, 64, 1, 1),
            ],
            config(false),
        ),
        ArenaErrorCode::DuplicateId,
    );
    assert_error_code!(
        plan_static_arena(&[lifetime("zero-size", 0, 64, 0, 0)], config(false)),
        ArenaErrorCode::ZeroByteSize,
    );
    assert_error_code!(
        plan_static_arena(&[lifetime("zero-align", 64, 0, 0, 0)], config(false)),
        ArenaErrorCode::ZeroTensorAlignment,
    );
    assert_error_code!(
        plan_static_arena(&[lifetime("bad-align", 64, 24, 0, 0)], config(false)),
        ArenaErrorCode::NonPowerOfTwoTensorAlignment,
    );
    assert_error_code!(
        plan_static_arena(&[lifetime("bad-range", 64, 64, 2, 1)], config(false)),
        ArenaErrorCode::ReversedLifetime,
    );
    assert_error_code!(
        plan_static_arena(
            &[lifetime("ok", 64, 64, 0, 0)],
            aligned_config(false, 0, 64),
        ),
        ArenaErrorCode::ZeroBaseAlignment,
    );
    assert_error_code!(
        plan_static_arena(
            &[lifetime("ok", 64, 64, 0, 0)],
            aligned_config(false, 64, 0),
        ),
        ArenaErrorCode::ZeroArenaAlignment,
    );
    assert_error_code!(
        plan_static_arena(
            &[lifetime("ok", 64, 64, 0, 0)],
            aligned_config(false, 24, 64),
        ),
        ArenaErrorCode::NonPowerOfTwoBaseAlignment,
    );
    assert_error_code!(
        plan_static_arena(
            &[lifetime("ok", 64, 64, 0, 0)],
            aligned_config(false, 64, 96),
        ),
        ArenaErrorCode::NonPowerOfTwoArenaAlignment,
    );
}

#[test]
fn rejects_offset_end_and_arena_overflow_with_u64_bounds() {
    let overflowing = vec![
        lifetime("huge-a", u64::MAX - 63, 64, 0, 0),
        lifetime("huge-b", 128, 64, 1, 1),
    ];
    assert_error_code!(
        plan_static_arena(&overflowing, config(false)),
        ArenaErrorCode::ArithmeticOverflow,
    );

    let end_overflow = vec![lifetime("end-overflow", u64::MAX, 1, 0, 0)];
    assert_error_code!(
        plan_static_arena(&end_overflow, config(false)),
        ArenaErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn verifier_rejects_corrupted_metadata_with_exact_error_codes() {
    let lifetimes = vec![lifetime("tensor", 128, 64, 0, 0)];
    let good_plan = plan_static_arena(&lifetimes, config(false)).unwrap();
    verify_static_arena_plan(&lifetimes, &good_plan, config(false)).unwrap();

    for corrupted in [
        StaticArenaPlan {
            arena_bytes: good_plan.arena_bytes,
            allocations: vec![ArenaAllocation {
                id: "mutated".to_owned(),
                ..good_plan.allocations[0].clone()
            }],
        },
        StaticArenaPlan {
            arena_bytes: good_plan.arena_bytes,
            allocations: vec![ArenaAllocation {
                id: "renamed".to_owned(),
                ..good_plan.allocations[0].clone()
            }],
        },
        StaticArenaPlan {
            arena_bytes: good_plan.arena_bytes,
            allocations: vec![ArenaAllocation {
                size: 256,
                ..good_plan.allocations[0].clone()
            }],
        },
        StaticArenaPlan {
            arena_bytes: good_plan.arena_bytes,
            allocations: vec![ArenaAllocation {
                alignment: 128,
                ..good_plan.allocations[0].clone()
            }],
        },
        StaticArenaPlan {
            arena_bytes: good_plan.arena_bytes,
            allocations: vec![ArenaAllocation {
                first_write: 1,
                ..good_plan.allocations[0].clone()
            }],
        },
        StaticArenaPlan {
            arena_bytes: good_plan.arena_bytes,
            allocations: vec![ArenaAllocation {
                last_use: 1,
                ..good_plan.allocations[0].clone()
            }],
        },
        StaticArenaPlan {
            arena_bytes: good_plan.arena_bytes,
            allocations: vec![ArenaAllocation {
                stage_label: Some("mutated".to_owned()),
                ..good_plan.allocations[0].clone()
            }],
        },
    ] {
        assert_error_code!(
            verify_static_arena_plan(&lifetimes, &corrupted, config(false)),
            ArenaErrorCode::MetadataMismatch,
        );
    }
}

#[test]
fn verifier_rejects_overlap_misalignment_out_of_bounds_duplicate_and_missing_allocations() {
    let lifetimes = vec![
        lifetime("left", 128, 64, 0, 0),
        lifetime("right", 128, 64, 1, 1),
    ];
    let good_plan = plan_static_arena(&lifetimes, config(true)).unwrap();
    verify_static_arena_plan(&lifetimes, &good_plan, config(true)).unwrap();

    let overlap = StaticArenaPlan {
        arena_bytes: 256,
        allocations: vec![
            ArenaAllocation {
                id: "left".to_owned(),
                offset: 0,
                size: 128,
                alignment: 64,
                first_write: 0,
                last_use: 0,
                stage_label: None,
            },
            ArenaAllocation {
                id: "right".to_owned(),
                offset: 64,
                size: 128,
                alignment: 64,
                first_write: 1,
                last_use: 1,
                stage_label: None,
            },
        ],
    };
    assert_error_code!(
        verify_static_arena_plan(&lifetimes, &overlap, config(false)),
        ArenaErrorCode::Overlap,
    );

    let misaligned = StaticArenaPlan {
        arena_bytes: 256,
        allocations: vec![
            ArenaAllocation {
                id: "left".to_owned(),
                offset: 8,
                size: 128,
                alignment: 64,
                first_write: 0,
                last_use: 0,
                stage_label: None,
            },
            good_plan.allocations[1].clone(),
        ],
    };
    assert_error_code!(
        verify_static_arena_plan(&lifetimes, &misaligned, config(false)),
        ArenaErrorCode::Misalignment,
    );

    let out_of_bounds = StaticArenaPlan {
        arena_bytes: 64,
        allocations: good_plan.allocations.clone(),
    };
    assert_error_code!(
        verify_static_arena_plan(&lifetimes, &out_of_bounds, config(true)),
        ArenaErrorCode::OutOfBounds,
    );

    let duplicate = StaticArenaPlan {
        arena_bytes: good_plan.arena_bytes,
        allocations: vec![
            good_plan.allocations[0].clone(),
            good_plan.allocations[0].clone(),
        ],
    };
    assert_error_code!(
        verify_static_arena_plan(&lifetimes, &duplicate, config(true)),
        ArenaErrorCode::DuplicateAllocation,
    );

    let missing = StaticArenaPlan {
        arena_bytes: good_plan.arena_bytes,
        allocations: vec![good_plan.allocations[0].clone()],
    };
    assert_error_code!(
        verify_static_arena_plan(&lifetimes, &missing, config(true)),
        ArenaErrorCode::MissingAllocation,
    );
}

proptest! {
    #[test]
    fn property_live_ranges_are_disjoint_aligned_bounded_and_alias_never_exceeds_no_alias(
        raw in proptest::collection::vec(
            (
                1_u64..2048,
                prop_oneof![Just(16_u64), Just(32_u64), Just(64_u64), Just(128_u64)],
                0_u32..8,
                0_u32..8
            ),
            1..12
        )
    ) {
        let lifetimes = raw
            .into_iter()
            .enumerate()
            .map(|(index, (size, alignment, a, b))| {
                let first_write = a.min(b);
                let last_use = a.max(b);
                lifetime(&format!("tensor_{index:02}"), size, alignment, first_write, last_use)
            })
            .collect::<Vec<_>>();

        let alias_plan = plan_static_arena(&lifetimes, config(true)).unwrap();
        let no_alias_plan = plan_static_arena(&lifetimes, config(false)).unwrap();
        verify_static_arena_plan(&lifetimes, &alias_plan, config(true)).unwrap();
        verify_static_arena_plan(&lifetimes, &no_alias_plan, config(false)).unwrap();
        assert_aligned_bounded_and_disjoint(&lifetimes, &alias_plan, config(true));
        assert_aligned_bounded_and_disjoint(&lifetimes, &no_alias_plan, config(false));
        prop_assert!(alias_plan.arena_bytes <= no_alias_plan.arena_bytes);
    }
}

#[test]
fn specialized_vision_layer_schedule_retains_exact_stage_labels_and_hits_liveness_lower_bound() {
    let lifetimes = vision_layer_schedule();
    let no_alias_plan = plan_static_arena(&lifetimes, config(false)).unwrap();
    let alias_plan = plan_static_arena(&lifetimes, config(true)).unwrap();

    verify_static_arena_plan(&lifetimes, &no_alias_plan, config(false)).unwrap();
    verify_static_arena_plan(&lifetimes, &alias_plan, config(true)).unwrap();
    assert_aligned_bounded_and_disjoint(&lifetimes, &no_alias_plan, config(false));
    assert_aligned_bounded_and_disjoint(&lifetimes, &alias_plan, config(true));

    assert_eq!(no_alias_plan.arena_bytes, VISION_NO_ALIAS_BYTES);
    // Contract: the deterministic planner must achieve the exact offline lower
    // bound for this 12-dispatch schedule, equivalent to size-desc first-fit
    // or any policy with the same packing result.
    assert_eq!(alias_plan.arena_bytes, VISION_ALIAS_LOWER_BOUND_BYTES);

    let retained_labels = alias_plan
        .allocations
        .iter()
        .map(|allocation| allocation.stage_label.as_deref().unwrap_or(""))
        .collect::<Vec<_>>();
    assert_eq!(
        retained_labels,
        vec![
            "input",
            "norm1",
            "query",
            "key",
            "value",
            "attention-context",
            "attention-output",
            "attention-residual",
            "norm2",
            "mlp-fc1",
            "mlp-activation",
            "mlp-output",
            "output",
        ]
    );
}
