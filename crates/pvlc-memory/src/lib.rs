//! Deterministic static-arena planner and verifier for buffer liveness.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArenaErrorCode {
    EmptyId,
    DuplicateId,
    ZeroByteSize,
    ZeroTensorAlignment,
    NonPowerOfTwoTensorAlignment,
    ReversedLifetime,
    ZeroBaseAlignment,
    ZeroArenaAlignment,
    NonPowerOfTwoBaseAlignment,
    NonPowerOfTwoArenaAlignment,
    ArithmeticOverflow,
    MetadataMismatch,
    Overlap,
    Misalignment,
    OutOfBounds,
    DuplicateAllocation,
    MissingAllocation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArenaError {
    code: ArenaErrorCode,
    message: &'static str,
}

impl ArenaError {
    #[must_use]
    pub const fn code(&self) -> ArenaErrorCode {
        self.code
    }

    const fn new(code: ArenaErrorCode, message: &'static str) -> Self {
        Self { code, message }
    }
}

impl fmt::Display for ArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "arena planner {:?}: {}", self.code, self.message)
    }
}

impl Error for ArenaError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaConfig {
    pub allow_aliasing: bool,
    pub arena_alignment: u64,
    pub base_alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorLifetime {
    pub id: String,
    pub byte_size: u64,
    pub alignment: u64,
    pub first_write: u32,
    pub last_use: u32,
    pub stage_label: Option<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ArenaAllocation {
    pub id: String,
    pub offset: u64,
    pub size: u64,
    pub alignment: u64,
    pub first_write: u32,
    pub last_use: u32,
    pub stage_label: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaPlan {
    pub arena_bytes: u64,
    pub allocations: Vec<ArenaAllocation>,
}

pub fn plan_static_arena(
    lifetimes: &[TensorLifetime],
    config: ArenaConfig,
) -> Result<StaticArenaPlan, ArenaError> {
    validate_config(config)?;
    validate_lifetimes(lifetimes)?;

    if lifetimes.is_empty() {
        return Ok(StaticArenaPlan {
            arena_bytes: 0,
            allocations: Vec::new(),
        });
    }

    let canonical = canonical_lifetimes(lifetimes);
    let no_alias_plan = plan_no_aliasing(&canonical, config)
        .and_then(|allocations| finalize_plan(allocations, config));
    if !config.allow_aliasing {
        return no_alias_plan;
    }

    let aliasing_plan = plan_aliasing_first_fit(&canonical, config)
        .and_then(|allocations| finalize_plan(allocations, config));
    match (aliasing_plan, no_alias_plan) {
        (Ok(aliasing), Ok(no_alias)) => {
            if aliasing.arena_bytes <= no_alias.arena_bytes {
                Ok(aliasing)
            } else {
                Ok(no_alias)
            }
        }
        (Ok(aliasing), Err(_)) => Ok(aliasing),
        (Err(_), Ok(no_alias)) => Ok(no_alias),
        (Err(error), Err(_)) => Err(error),
    }
}

pub fn verify_static_arena_plan(
    lifetimes: &[TensorLifetime],
    plan: &StaticArenaPlan,
    config: ArenaConfig,
) -> Result<(), ArenaError> {
    validate_config(config)?;
    validate_lifetimes(lifetimes)?;

    if lifetimes.is_empty() {
        if plan.arena_bytes == 0 && plan.allocations.is_empty() {
            return Ok(());
        }
        return Err(missing_allocation("empty input must produce an empty plan"));
    }

    if !plan.arena_bytes.is_multiple_of(config.arena_alignment) {
        return Err(out_of_bounds("arena size must respect arena alignment"));
    }

    let canonical = canonical_lifetimes(lifetimes);
    let mut seen_ids = HashSet::with_capacity(plan.allocations.len());
    for allocation in &plan.allocations {
        if !seen_ids.insert(allocation.id.as_str()) {
            return Err(duplicate_allocation(
                "plan contains duplicate allocation ids",
            ));
        }
    }
    if plan.allocations.len() != canonical.len() {
        return Err(missing_allocation(
            "plan must contain exactly one allocation per lifetime",
        ));
    }

    for (allocation, lifetime) in plan.allocations.iter().zip(canonical.iter()) {
        let expected_alignment = effective_alignment(config, lifetime);
        if allocation.id != lifetime.id
            || allocation.size != lifetime.byte_size
            || allocation.alignment != expected_alignment
            || allocation.first_write != lifetime.first_write
            || allocation.last_use != lifetime.last_use
            || allocation.stage_label != lifetime.stage_label
        {
            return Err(metadata_mismatch(
                "allocation metadata must match the canonical lifetime contract",
            ));
        }
        if allocation.offset % allocation.alignment != 0 {
            return Err(misalignment(
                "allocation offset must respect the stored allocation alignment",
            ));
        }
        if checked_add(allocation.offset, allocation.size)? > plan.arena_bytes {
            return Err(out_of_bounds("allocation exceeds the arena bounds"));
        }
    }

    for (index, left) in plan.allocations.iter().enumerate() {
        for right in &plan.allocations[index + 1..] {
            if !ranges_overlap(left.offset, left.size, right.offset, right.size)? {
                continue;
            }
            if !config.allow_aliasing
                || lifetimes_overlap(
                    left.first_write,
                    left.last_use,
                    right.first_write,
                    right.last_use,
                )
            {
                return Err(overlap(
                    "overlapping byte ranges are not allowed for these lifetimes",
                ));
            }
        }
    }

    Ok(())
}

fn validate_config(config: ArenaConfig) -> Result<(), ArenaError> {
    if config.base_alignment == 0 {
        return Err(ArenaError::new(
            ArenaErrorCode::ZeroBaseAlignment,
            "base alignment must be nonzero",
        ));
    }
    if config.arena_alignment == 0 {
        return Err(ArenaError::new(
            ArenaErrorCode::ZeroArenaAlignment,
            "arena alignment must be nonzero",
        ));
    }
    if !config.base_alignment.is_power_of_two() {
        return Err(ArenaError::new(
            ArenaErrorCode::NonPowerOfTwoBaseAlignment,
            "base alignment must be a power of two",
        ));
    }
    if !config.arena_alignment.is_power_of_two() {
        return Err(ArenaError::new(
            ArenaErrorCode::NonPowerOfTwoArenaAlignment,
            "arena alignment must be a power of two",
        ));
    }
    Ok(())
}

fn validate_lifetimes(lifetimes: &[TensorLifetime]) -> Result<(), ArenaError> {
    let mut ids = HashSet::with_capacity(lifetimes.len());
    for lifetime in lifetimes {
        if lifetime.id.is_empty() {
            return Err(ArenaError::new(
                ArenaErrorCode::EmptyId,
                "lifetime id must be nonempty",
            ));
        }
        if !ids.insert(lifetime.id.as_str()) {
            return Err(ArenaError::new(
                ArenaErrorCode::DuplicateId,
                "lifetime ids must be unique",
            ));
        }
        if lifetime.byte_size == 0 {
            return Err(ArenaError::new(
                ArenaErrorCode::ZeroByteSize,
                "lifetime byte size must be nonzero",
            ));
        }
        if lifetime.alignment == 0 {
            return Err(ArenaError::new(
                ArenaErrorCode::ZeroTensorAlignment,
                "tensor alignment must be nonzero",
            ));
        }
        if !lifetime.alignment.is_power_of_two() {
            return Err(ArenaError::new(
                ArenaErrorCode::NonPowerOfTwoTensorAlignment,
                "tensor alignment must be a power of two",
            ));
        }
        if lifetime.first_write > lifetime.last_use {
            return Err(ArenaError::new(
                ArenaErrorCode::ReversedLifetime,
                "lifetime range must be ordered",
            ));
        }
    }
    Ok(())
}

fn canonical_lifetimes(lifetimes: &[TensorLifetime]) -> Vec<&TensorLifetime> {
    let mut canonical = lifetimes.iter().collect::<Vec<_>>();
    canonical.sort_by(|left, right| {
        (left.first_write, left.id.as_str()).cmp(&(right.first_write, right.id.as_str()))
    });
    canonical
}

fn effective_alignment(config: ArenaConfig, lifetime: &TensorLifetime) -> u64 {
    config.base_alignment.max(lifetime.alignment)
}

fn plan_no_aliasing(
    canonical: &[&TensorLifetime],
    config: ArenaConfig,
) -> Result<Vec<ArenaAllocation>, ArenaError> {
    let mut placed = Vec::with_capacity(canonical.len());
    for lifetime in canonical {
        let alignment = effective_alignment(config, lifetime);
        let offset = lowest_available_offset(lifetime, alignment, &placed, false)?;
        placed.push(allocation_for(lifetime, offset, alignment));
    }
    Ok(placed)
}

fn plan_aliasing_first_fit(
    canonical: &[&TensorLifetime],
    config: ArenaConfig,
) -> Result<Vec<ArenaAllocation>, ArenaError> {
    let mut placement_order = canonical.to_vec();
    placement_order.sort_by(|left, right| {
        right
            .byte_size
            .cmp(&left.byte_size)
            .then_with(|| {
                effective_alignment(config, right).cmp(&effective_alignment(config, left))
            })
            .then_with(|| left.first_write.cmp(&right.first_write))
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut placed = Vec::with_capacity(canonical.len());
    for lifetime in placement_order {
        let alignment = effective_alignment(config, lifetime);
        let offset = lowest_available_offset(lifetime, alignment, &placed, true)?;
        placed.push(allocation_for(lifetime, offset, alignment));
    }
    Ok(placed)
}

fn lowest_available_offset(
    lifetime: &TensorLifetime,
    alignment: u64,
    placed: &[ArenaAllocation],
    allow_lifetime_aliasing: bool,
) -> Result<u64, ArenaError> {
    let mut conflicts = placed
        .iter()
        .filter(|allocation| {
            !allow_lifetime_aliasing
                || lifetimes_overlap(
                    lifetime.first_write,
                    lifetime.last_use,
                    allocation.first_write,
                    allocation.last_use,
                )
        })
        .collect::<Vec<_>>();
    conflicts.sort_by(|left, right| {
        (left.offset, left.size, left.id.as_str()).cmp(&(
            right.offset,
            right.size,
            right.id.as_str(),
        ))
    });

    let mut candidate = 0_u64;
    for allocation in conflicts {
        candidate = align_up(candidate, alignment)?;
        let candidate_end = checked_add(candidate, lifetime.byte_size)?;
        if candidate_end <= allocation.offset {
            return Ok(candidate);
        }
        let allocation_end = checked_add(allocation.offset, allocation.size)?;
        if candidate < allocation_end {
            candidate = allocation_end;
        }
    }

    candidate = align_up(candidate, alignment)?;
    checked_add(candidate, lifetime.byte_size)?;
    Ok(candidate)
}

fn allocation_for(lifetime: &TensorLifetime, offset: u64, alignment: u64) -> ArenaAllocation {
    ArenaAllocation {
        id: lifetime.id.clone(),
        offset,
        size: lifetime.byte_size,
        alignment,
        first_write: lifetime.first_write,
        last_use: lifetime.last_use,
        stage_label: lifetime.stage_label.clone(),
    }
}

fn finalize_plan(
    mut allocations: Vec<ArenaAllocation>,
    config: ArenaConfig,
) -> Result<StaticArenaPlan, ArenaError> {
    allocations.sort_by(|left, right| {
        (left.first_write, left.id.as_str()).cmp(&(right.first_write, right.id.as_str()))
    });
    let arena_high_water = allocations.iter().try_fold(0_u64, |peak, allocation| {
        Ok(peak.max(checked_add(allocation.offset, allocation.size)?))
    })?;
    Ok(StaticArenaPlan {
        arena_bytes: align_up(arena_high_water, config.arena_alignment)?,
        allocations,
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ArenaError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|rounded| rounded & !mask)
        .ok_or_else(arithmetic_overflow)
}

fn checked_add(left: u64, right: u64) -> Result<u64, ArenaError> {
    left.checked_add(right).ok_or_else(arithmetic_overflow)
}

fn ranges_overlap(
    left_offset: u64,
    left_size: u64,
    right_offset: u64,
    right_size: u64,
) -> Result<bool, ArenaError> {
    let left_end = checked_add(left_offset, left_size)?;
    let right_end = checked_add(right_offset, right_size)?;
    Ok(left_offset < right_end && right_offset < left_end)
}

fn lifetimes_overlap(left_first: u32, left_last: u32, right_first: u32, right_last: u32) -> bool {
    left_first <= right_last && right_first <= left_last
}

fn arithmetic_overflow() -> ArenaError {
    ArenaError::new(
        ArenaErrorCode::ArithmeticOverflow,
        "arena arithmetic overflowed u64 bounds",
    )
}

fn metadata_mismatch(message: &'static str) -> ArenaError {
    ArenaError::new(ArenaErrorCode::MetadataMismatch, message)
}

fn overlap(message: &'static str) -> ArenaError {
    ArenaError::new(ArenaErrorCode::Overlap, message)
}

fn misalignment(message: &'static str) -> ArenaError {
    ArenaError::new(ArenaErrorCode::Misalignment, message)
}

fn out_of_bounds(message: &'static str) -> ArenaError {
    ArenaError::new(ArenaErrorCode::OutOfBounds, message)
}

fn duplicate_allocation(message: &'static str) -> ArenaError {
    ArenaError::new(ArenaErrorCode::DuplicateAllocation, message)
}

fn missing_allocation(message: &'static str) -> ArenaError {
    ArenaError::new(ArenaErrorCode::MissingAllocation, message)
}
