//! Host-only contract for the opt-in vision-stack poison/canary memory plan.
//!
//! The browser runtime consumes this plan, but its arithmetic and failure modes
//! must remain independently testable without wasm, WebGPU, or a browser.

use std::str::FromStr;

use pvlc_runtime_web::{
    VISION_STACK_PREFIX_CANARY_U32, VISION_STACK_SCRATCH_POISON_U32,
    VISION_STACK_SUFFIX_CANARY_U32, VisionStackMemoryHardening, VisionStackMemoryHardeningPlan,
};

fn plan(
    alignment: u64,
    logical_scratch_bytes: u64,
    semantic_checkpoint_bytes: u64,
    logical_peak_gpu_data_bytes: u64,
) -> VisionStackMemoryHardeningPlan {
    VisionStackMemoryHardeningPlan::new(
        VisionStackMemoryHardening::PoisonCanary,
        alignment,
        logical_scratch_bytes,
        semantic_checkpoint_bytes,
        logical_peak_gpu_data_bytes,
    )
    .expect("valid poison/canary plan")
}

fn guarded_readback(plan: &VisionStackMemoryHardeningPlan) -> Vec<u8> {
    let semantic_len = usize::try_from(plan.semantic_checkpoint_bytes()).unwrap();
    let guard_words = usize::try_from(plan.guard_bytes() / 4).unwrap();
    let mut mapped = (0..semantic_len)
        .map(|index| ((index * 37 + 11) % 251) as u8)
        .collect::<Vec<_>>();
    for pattern in [
        VISION_STACK_PREFIX_CANARY_U32,
        VISION_STACK_SUFFIX_CANARY_U32,
    ] {
        for _ in 0..guard_words {
            mapped.extend_from_slice(&pattern.to_le_bytes());
        }
    }
    assert_eq!(
        u64::try_from(mapped.len()).unwrap(),
        plan.physical_readback_bytes(),
    );
    mapped
}

#[test]
fn poison_canary_mode_and_patterns_are_stable_and_fail_closed() {
    assert_eq!(VISION_STACK_SCRATCH_POISON_U32, 0x7fc0_a5a5);
    assert_eq!(VISION_STACK_PREFIX_CANARY_U32, 0x51c0_ffee);
    assert_eq!(VISION_STACK_SUFFIX_CANARY_U32, 0xa11a_5eed);

    let mode = VisionStackMemoryHardening::from_str("poison_canary")
        .expect("the reviewed opt-in mode must parse");
    assert_eq!(mode, VisionStackMemoryHardening::PoisonCanary);
    assert_eq!(mode.as_str(), "poison_canary");
    for invalid in [
        "",
        "none",
        "poison",
        "canary",
        "separate_buffers",
        "POISON_CANARY",
        "poison_canary ",
    ] {
        assert!(
            VisionStackMemoryHardening::from_str(invalid).is_err(),
            "unexpected memory-hardening mode {invalid:?} was accepted",
        );
    }
}

#[test]
fn compact_alignment_256_plan_uses_exact_symmetric_guards_and_physical_bytes() {
    let plan = plan(256, 1_024, 144, 1_844);
    assert_eq!(plan.mode(), VisionStackMemoryHardening::PoisonCanary);
    assert_eq!(plan.storage_alignment(), 256);
    assert_eq!(plan.guard_bytes(), 256);

    assert_eq!(plan.logical_scratch_bytes(), 1_024);
    assert_eq!(plan.scratch_logical_offset(), 256);
    assert_eq!(plan.scratch_suffix_offset(), 1_280);
    assert_eq!(plan.physical_scratch_bytes(), 1_536);

    assert_eq!(plan.semantic_checkpoint_bytes(), 144);
    assert_eq!(plan.readback_prefix_canary_offset(), 144);
    assert_eq!(plan.readback_suffix_canary_offset(), 400);
    assert_eq!(plan.physical_readback_bytes(), 656);

    assert_eq!(plan.logical_peak_gpu_data_bytes(), 1_844);
    assert_eq!(plan.physical_peak_gpu_data_bytes(), 2_868);
}

#[test]
fn compact_alignment_32_and_portable_alignment_64_plans_do_not_assume_chrome() {
    let webkit = plan(32, 256, 144, 1_076);
    assert_eq!(webkit.guard_bytes(), 32);
    assert_eq!(webkit.physical_scratch_bytes(), 320);
    assert_eq!(webkit.physical_readback_bytes(), 208);
    assert_eq!(webkit.physical_peak_gpu_data_bytes(), 1_204);

    let portable = plan(64, 256, 144, 1_076);
    assert_eq!(portable.guard_bytes(), 64);
    assert_eq!(portable.physical_scratch_bytes(), 384);
    assert_eq!(portable.physical_readback_bytes(), 272);
    assert_eq!(portable.physical_peak_gpu_data_bytes(), 1_332);

    for alignment in [4, 8, 16] {
        let minimum = plan(alignment, 16, 4, 20);
        assert_eq!(minimum.guard_bytes(), alignment);
        assert_eq!(minimum.physical_scratch_bytes(), 16 + 2 * alignment);
        assert_eq!(minimum.physical_readback_bytes(), 4 + 2 * alignment);
        assert_eq!(minimum.physical_peak_gpu_data_bytes(), 20 + 4 * alignment);
    }
}

#[test]
fn scratch_binding_offsets_shift_by_one_guard_without_changing_semantic_ranges() {
    let plan = plan(256, 1_024, 144, 1_844);
    for (logical_offset, bytes, physical_offset) in [
        (0, 48, 256),
        (256, 48, 512),
        (512, 60, 768),
        (768, 48, 1_024),
        (768, 256, 1_024),
    ] {
        let binding = plan
            .shift_scratch_binding(logical_offset, bytes)
            .expect("canonical logical binding must shift into the guarded arena");
        assert_eq!(binding.logical_offset(), logical_offset);
        assert_eq!(binding.physical_offset(), physical_offset);
        assert_eq!(binding.bytes(), bytes);
        assert!(binding.physical_offset() >= plan.guard_bytes());
        assert!(binding.physical_offset() + binding.bytes() <= plan.scratch_suffix_offset(),);
    }
}

#[test]
fn mapped_readback_verification_returns_only_the_borrowed_semantic_prefix() {
    let plan = plan(256, 1_024, 144, 1_844);
    let mapped = guarded_readback(&plan);
    let semantic = plan
        .verify_and_split_readback(&mapped)
        .expect("exact little-endian canaries must verify");
    assert_eq!(semantic, &mapped[..144]);
    assert_eq!(semantic.len(), 144);
    assert!(std::ptr::eq(semantic.as_ptr(), mapped.as_ptr()));
}

#[test]
fn mapped_readback_verification_rejects_first_middle_and_last_word_corruption_in_both_guards() {
    let plan = plan(256, 1_024, 144, 1_844);
    let prefix = usize::try_from(plan.readback_prefix_canary_offset()).unwrap();
    let suffix = usize::try_from(plan.readback_suffix_canary_offset()).unwrap();
    let guard = usize::try_from(plan.guard_bytes()).unwrap();
    for (label, guard_start) in [("prefix", prefix), ("suffix", suffix)] {
        for word in [0, guard / 8, guard / 4 - 1] {
            let mut corrupted = guarded_readback(&plan);
            corrupted[guard_start + word * 4] ^= 0x01;
            assert!(
                plan.verify_and_split_readback(&corrupted).is_err(),
                "{label} guard corruption at word {word} was accepted",
            );
        }
    }
}

#[test]
fn mapped_readback_verification_rejects_every_wrong_physical_length() {
    let plan = plan(32, 256, 144, 1_076);
    let mapped = guarded_readback(&plan);
    for wrong in [
        mapped[..mapped.len() - 1].to_vec(),
        mapped[..mapped.len() - 4].to_vec(),
        {
            let mut oversized = mapped.clone();
            oversized.push(0);
            oversized
        },
        mapped[..144].to_vec(),
    ] {
        assert!(
            plan.verify_and_split_readback(&wrong).is_err(),
            "wrong mapped readback length {} was accepted",
            wrong.len(),
        );
    }
}

#[test]
fn invalid_alignment_sizes_and_binding_ranges_are_rejected() {
    for alignment in [0, 1, 2, 3, 6, 12, 48, 96] {
        assert!(
            VisionStackMemoryHardeningPlan::new(
                VisionStackMemoryHardening::PoisonCanary,
                alignment,
                1_024,
                144,
                1_844,
            )
            .is_err(),
            "invalid storage alignment {alignment} was accepted",
        );
    }

    for (scratch, checkpoint, peak) in [
        (0, 144, 1_844),
        (1_022, 144, 1_844),
        (1_024, 0, 1_844),
        (1_024, 142, 1_844),
        (1_024, 144, 0),
        (1_024, 144, 1_020),
    ] {
        assert!(
            VisionStackMemoryHardeningPlan::new(
                VisionStackMemoryHardening::PoisonCanary,
                256,
                scratch,
                checkpoint,
                peak,
            )
            .is_err(),
            "invalid byte geometry {scratch}/{checkpoint}/{peak} was accepted",
        );
    }

    let plan = plan(256, 1_024, 144, 1_844);
    for (offset, bytes) in [
        (0, 0),
        (0, 2),
        (1, 48),
        (128, 48),
        (1_024, 4),
        (768, 260),
        (u64::MAX - 255, 256),
    ] {
        assert!(
            plan.shift_scratch_binding(offset, bytes).is_err(),
            "invalid logical scratch binding {offset}+{bytes} was accepted",
        );
    }
}

#[test]
fn all_guard_and_total_byte_arithmetic_is_checked_for_overflow() {
    let largest_multiple_of_four = u64::MAX - 3;
    for (alignment, scratch, checkpoint, peak) in [
        (1_u64 << 63, 4, 4, 4),
        (4, largest_multiple_of_four, 4, largest_multiple_of_four),
        (4, 4, largest_multiple_of_four, largest_multiple_of_four),
        (4, 4, 4, largest_multiple_of_four),
    ] {
        assert!(
            VisionStackMemoryHardeningPlan::new(
                VisionStackMemoryHardening::PoisonCanary,
                alignment,
                scratch,
                checkpoint,
                peak,
            )
            .is_err(),
            "overflowing plan {alignment}/{scratch}/{checkpoint}/{peak} was accepted",
        );
    }
}
