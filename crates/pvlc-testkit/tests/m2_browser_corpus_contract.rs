use std::collections::{BTreeMap, BTreeSet};

use pvlc_runtime_core::{KernelId, KernelInvocation};
use pvlc_testkit::{
    ComparisonAxes, M2_BOUNDARIES, M2_INPUT_FAMILIES, M2PrimitiveCase, M2PrimitiveCorpus,
    compare_f32, m2_primitive_corpus,
};

const EXPECTED_KERNELS: [KernelId; 7] = [
    KernelId::GemmF32,
    KernelId::GemvF32,
    KernelId::LayerNormF32,
    KernelId::RmsNormF32,
    KernelId::SiluF32,
    KernelId::GeluTanhF32,
    KernelId::RopeNeoxF32,
];
const EXPECTED_BOUNDARIES: [u32; 25] = [
    1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
    1023, 1024,
];
const EXPECTED_INPUT_FAMILIES: [&str; 8] = [
    "zeros",
    "ones",
    "alternating-signs",
    "tiny",
    "near-fp16-limit",
    "impulse",
    "repeated-pattern",
    "random",
];

fn has_tag(case: &M2PrimitiveCase, tag: &str) -> bool {
    case.tags.iter().any(|candidate| candidate == tag)
}

fn cases_for(
    corpus: &M2PrimitiveCorpus,
    kernel: KernelId,
) -> impl Iterator<Item = &M2PrimitiveCase> {
    corpus
        .cases
        .iter()
        .filter(move |case| case.invocation.kernel_id() == kernel)
}

fn assert_boundary_axis<F>(
    corpus: &M2PrimitiveCorpus,
    kernel: KernelId,
    axis_name: &str,
    dimension: F,
) where
    F: Fn(&KernelInvocation) -> Option<u32>,
{
    for boundary in M2_BOUNDARIES {
        assert!(
            cases_for(corpus, kernel).any(|case| {
                has_tag(case, "boundary") && dimension(&case.invocation) == Some(boundary)
            }),
            "{kernel} has no {axis_name} boundary case for {boundary}"
        );
    }
}

fn family_operand(case: &M2PrimitiveCase) -> (&str, &[f32]) {
    let operand = case
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix("family-operand:"))
        .unwrap_or_else(|| panic!("{} has no family-operand tag", case.id));
    let values = match (&case.invocation, operand) {
        (KernelInvocation::GemmF32 { left, .. }, "left") => left,
        (KernelInvocation::GemmF32 { right, .. }, "right") => right,
        (KernelInvocation::GemvF32 { matrix, .. }, "matrix") => matrix,
        (KernelInvocation::GemvF32 { vector, .. }, "vector") => vector,
        (KernelInvocation::LayerNormF32 { input, .. }, "input")
        | (KernelInvocation::RmsNormF32 { input, .. }, "input") => input,
        (KernelInvocation::SiluF32 { values }, "values")
        | (KernelInvocation::GeluTanhF32 { values }, "values")
        | (KernelInvocation::RopeNeoxF32 { values, .. }, "values") => values,
        _ => panic!("{} has invalid family operand {operand}", case.id),
    };
    (operand, values)
}

fn assert_family_pattern(family: &str, values: &[f32], case_id: &str) {
    assert!(
        values.len() >= 8,
        "{case_id} family operand is too small to be meaningful"
    );
    match family {
        "zeros" => assert!(values.iter().all(|value| *value == 0.0), "{case_id}"),
        "ones" => assert!(values.iter().all(|value| *value == 1.0), "{case_id}"),
        "alternating-signs" => assert!(
            values.iter().enumerate().all(|(index, value)| {
                *value == if index.is_multiple_of(2) { -1.0 } else { 1.0 }
            }),
            "{case_id}"
        ),
        "tiny" => assert!(
            values
                .iter()
                .all(|value| value.abs() > 0.0 && value.abs() <= 1.0e-29),
            "{case_id}"
        ),
        "near-fp16-limit" => assert!(
            values.iter().all(|value| value.abs() == 65_504.0),
            "{case_id}"
        ),
        "impulse" => assert_eq!(
            values
                .iter()
                .filter(|value| **value != 0.0)
                .copied()
                .collect::<Vec<_>>(),
            [1.0],
            "{case_id}"
        ),
        "repeated-pattern" => {
            assert_ne!(values[0], values[1], "{case_id}");
            assert!(
                values
                    .iter()
                    .enumerate()
                    .all(|(index, value)| *value == values[index % 3]),
                "{case_id}"
            );
        }
        "random" => {
            let distinct: BTreeSet<_> = values.iter().map(|value| value.to_bits()).collect();
            assert!(distinct.len() >= 8, "{case_id}");
            assert!(values.iter().any(|value| *value < 0.0), "{case_id}");
            assert!(values.iter().any(|value| *value > 0.0), "{case_id}");
        }
        _ => panic!("unknown independent family {family}"),
    }
}

#[test]
fn corpus_is_deterministic_strictly_serializable_and_self_consistent() {
    assert_eq!(KernelId::M2_PRIMITIVES, EXPECTED_KERNELS);
    assert_eq!(M2_BOUNDARIES, EXPECTED_BOUNDARIES);
    assert_eq!(M2_INPUT_FAMILIES, EXPECTED_INPUT_FAMILIES);

    let first = m2_primitive_corpus().unwrap();
    let second = m2_primitive_corpus().unwrap();
    assert_eq!(
        first, second,
        "the checked browser corpus must be deterministic"
    );
    assert_eq!(first.schema_version, 1);
    assert_eq!(first.oracle, "pvlc-cpu-ref/f32-v1");
    assert!(
        first.cases.len() >= 300,
        "M2b must exercise the full boundary/family corpus, not a seven-case smoke test"
    );

    let canonical = serde_json::to_vec(&first).unwrap();
    let roundtrip: M2PrimitiveCorpus = serde_json::from_slice(&canonical).unwrap();
    assert_eq!(roundtrip, first);
    assert_eq!(
        blake3::hash(&canonical),
        blake3::hash(&serde_json::to_vec(&second).unwrap())
    );

    let mut ids = BTreeSet::new();
    for case in &first.cases {
        assert!(!case.id.is_empty());
        assert!(
            ids.insert(case.id.as_str()),
            "duplicate case id: {}",
            case.id
        );
        assert_eq!(case.kernel, case.invocation.kernel_id());
        assert!(!case.tags.is_empty(), "{} has no coverage tags", case.id);
        assert!(
            case.tags.windows(2).all(|pair| pair[0] < pair[1]),
            "{} tags must be sorted and unique",
            case.id
        );

        let plan = case.invocation.plan().unwrap();
        assert_eq!(case.expected.len(), plan.output_elements, "{}", case.id);
        assert_eq!(
            case.shape.iter().product::<usize>(),
            plan.output_elements,
            "{}",
            case.id
        );
        assert!(
            case.expected.iter().all(|value| value.is_finite()),
            "{}",
            case.id
        );

        let identity = compare_f32(
            &case.expected,
            &case.expected,
            &case.shape,
            ComparisonAxes::default(),
        )
        .unwrap();
        assert!(
            identity
                .assess(&case.policy.comparison_policy())
                .unwrap()
                .passed(),
            "{} carries an unusable comparison policy",
            case.id
        );
    }

    let kernels: BTreeSet<_> = first.cases.iter().map(|case| case.kernel).collect();
    assert_eq!(kernels, EXPECTED_KERNELS.into_iter().collect());
}

#[test]
fn corpus_covers_every_m2a_boundary_axis() {
    let corpus = m2_primitive_corpus().unwrap();

    assert_boundary_axis(
        &corpus,
        KernelId::GemmF32,
        "rows",
        |invocation| match invocation {
            KernelInvocation::GemmF32 { rows, .. } => Some(*rows),
            _ => None,
        },
    );
    assert_boundary_axis(
        &corpus,
        KernelId::GemmF32,
        "inner",
        |invocation| match invocation {
            KernelInvocation::GemmF32 { inner, .. } => Some(*inner),
            _ => None,
        },
    );
    assert_boundary_axis(
        &corpus,
        KernelId::GemmF32,
        "columns",
        |invocation| match invocation {
            KernelInvocation::GemmF32 { columns, .. } => Some(*columns),
            _ => None,
        },
    );
    assert_boundary_axis(
        &corpus,
        KernelId::GemvF32,
        "rows",
        |invocation| match invocation {
            KernelInvocation::GemvF32 { rows, .. } => Some(*rows),
            _ => None,
        },
    );
    assert_boundary_axis(
        &corpus,
        KernelId::GemvF32,
        "columns",
        |invocation| match invocation {
            KernelInvocation::GemvF32 { columns, .. } => Some(*columns),
            _ => None,
        },
    );
    for kernel in [KernelId::LayerNormF32, KernelId::RmsNormF32] {
        assert_boundary_axis(&corpus, kernel, "rows", |invocation| match invocation {
            KernelInvocation::LayerNormF32 { rows, .. }
            | KernelInvocation::RmsNormF32 { rows, .. } => Some(*rows),
            _ => None,
        });
        assert_boundary_axis(&corpus, kernel, "width", |invocation| match invocation {
            KernelInvocation::LayerNormF32 { width, .. }
            | KernelInvocation::RmsNormF32 { width, .. } => Some(*width),
            _ => None,
        });
    }
    for kernel in [KernelId::SiluF32, KernelId::GeluTanhF32] {
        assert_boundary_axis(&corpus, kernel, "length", |invocation| match invocation {
            KernelInvocation::SiluF32 { values } | KernelInvocation::GeluTanhF32 { values } => {
                u32::try_from(values.len()).ok()
            }
            _ => None,
        });
    }
    assert_boundary_axis(
        &corpus,
        KernelId::RopeNeoxF32,
        "rows",
        |invocation| match invocation {
            KernelInvocation::RopeNeoxF32 { rows, .. } => Some(*rows),
            _ => None,
        },
    );
}

#[test]
fn corpus_covers_all_input_families_for_every_kernel_and_rope_invariants() {
    let corpus = m2_primitive_corpus().unwrap();
    for kernel in EXPECTED_KERNELS {
        for family in EXPECTED_INPUT_FAMILIES {
            let tag = format!("family:{family}");
            let matches = cases_for(&corpus, kernel)
                .filter(|case| has_tag(case, &tag))
                .collect::<Vec<_>>();
            assert!(
                !matches.is_empty(),
                "{kernel} has no {family} input-family case"
            );
            for case in matches {
                let (_, values) = family_operand(case);
                assert_family_pattern(family, values, &case.id);
            }
        }
    }

    let mut rotary_dimensions = BTreeSet::new();
    let mut bases = BTreeSet::new();
    let mut observed_positions = BTreeSet::new();
    for case in cases_for(&corpus, KernelId::RopeNeoxF32) {
        let KernelInvocation::RopeNeoxF32 {
            rows,
            width,
            rotary_dim,
            positions,
            base,
            values,
        } = &case.invocation
        else {
            unreachable!()
        };
        rotary_dimensions.insert(*rotary_dim);
        bases.insert(base.to_bits());
        observed_positions.extend(positions.iter().copied());

        for (row, position) in positions.iter().copied().enumerate().take(*rows as usize) {
            if position == 0 {
                let start = row * *width as usize;
                let end = start + *rotary_dim as usize;
                assert_eq!(
                    &case.expected[start..end],
                    &values[start..end],
                    "{}",
                    case.id
                );
            }
            let start = row * *width as usize + *rotary_dim as usize;
            let end = (row + 1) * *width as usize;
            assert_eq!(
                &case.expected[start..end],
                &values[start..end],
                "{}",
                case.id
            );
        }
    }
    assert!(rotary_dimensions.is_superset(&[2, 4, 8, 16, 32, 64].into_iter().collect()));
    assert!(
        bases.is_superset(
            &[2.0_f32, 10_000.0, 500_000.0, 1_000_000.0]
                .into_iter()
                .map(f32::to_bits)
                .collect()
        )
    );
    assert!(observed_positions.is_superset(&[0, 1, 17, 127, 4096].into_iter().collect()));
}

#[test]
fn every_kernel_has_independently_identified_cpu_and_native_comparison_budgets() {
    let corpus = m2_primitive_corpus().unwrap();
    let mut policies = BTreeMap::<KernelId, BTreeSet<(u64, u64)>>::new();
    for case in &corpus.cases {
        assert!(case.policy.max_abs.is_finite() && case.policy.max_abs > 0.0);
        assert!(case.policy.max_relative_l2.is_finite() && case.policy.max_relative_l2 > 0.0);
        assert!(case.policy.native_max_abs <= case.policy.max_abs);
        assert!(case.policy.native_max_relative_l2 <= case.policy.max_relative_l2);
        let (max_abs, max_relative_l2, native_max_abs, native_max_relative_l2) = match case.kernel {
            KernelId::GemmF32 | KernelId::GemvF32 => (1.0, 5.0e-5, 0.25, 3.0e-5),
            KernelId::LayerNormF32 | KernelId::RmsNormF32 => (2.5e-4, 1.0e-4, 2.0e-4, 8.0e-5),
            KernelId::SiluF32 | KernelId::GeluTanhF32 => (5.0e-2, 5.0e-5, 1.0e-2, 3.0e-5),
            KernelId::RopeNeoxF32 => (1.0, 2.0e-4, 0.1, 1.0e-4),
            KernelId::VisionAttentionF32
            | KernelId::VisionRope2dF32
            | KernelId::VisionPatchProjectionF32
            | KernelId::AddF32
            | KernelId::GeluErfF32
            | KernelId::ProjectorMerge2x2F32
            | KernelId::VisionQkvFusedF32
            | KernelId::DecoderKvAppendF32
            | KernelId::DecoderGqaF32
            | KernelId::DecoderGqaSplitPartialF32
            | KernelId::DecoderGqaSplitMergeF32
            | KernelId::DecoderMropeF32
            | KernelId::DecoderSwigluF32
            | KernelId::DecoderPrefillGqaF32
            | KernelId::DecoderPrefillMropeF32
            | KernelId::DecoderKvAppendRangeF32
            | KernelId::GemvTiledF32
            | KernelId::RmsNormF16Weights
            | KernelId::GemvTiledF16Weights
            | KernelId::LinearProjectionF16Weights
            | KernelId::VisionQkvFusedF16Weights
            | KernelId::LayerNormF16
            | KernelId::LinearProjectionF16
            | KernelId::VisionAttentionF16
            | KernelId::AddF16
            | KernelId::GeluTanhF16
            | KernelId::VisionRope2dF16
            | KernelId::ProjectorMerge2x2F16
            | KernelId::GeluErfF16 => {
                panic!("post-M2 kernel leaked into the frozen M2 browser corpus")
            }
        };
        assert!(case.policy.max_abs <= max_abs, "{} CPU max_abs", case.id);
        assert!(
            case.policy.max_relative_l2 <= max_relative_l2,
            "{} CPU relative_l2",
            case.id
        );
        assert!(
            case.policy.native_max_abs <= native_max_abs,
            "{} native max_abs",
            case.id
        );
        assert!(
            case.policy.native_max_relative_l2 <= native_max_relative_l2,
            "{} native relative_l2",
            case.id
        );
        policies.entry(case.kernel).or_default().insert((
            case.policy.max_abs.to_bits(),
            case.policy.native_max_abs.to_bits(),
        ));
    }
    assert_eq!(policies.len(), EXPECTED_KERNELS.len());
}
