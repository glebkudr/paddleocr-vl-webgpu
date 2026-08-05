use pvlc_testkit::{
    ComparisonAxes, ComparisonErrorCode, ComparisonPolicy, MetricViolation, StableTokenVerdict,
    compare_f32, compare_logits,
};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.12}, got {actual:.12}"
    );
}

fn permissive_policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 10.0,
        max_mean_abs: 10.0,
        max_p99_abs: 10.0,
        max_relative_l2: 10.0,
        min_cosine_similarity: -1.0,
        max_per_token_relative_l2: Some(10.0),
        max_per_channel_relative_l2: Some(10.0),
    }
}

#[test]
fn report_computes_every_required_metric_with_documented_quantiles_and_axes() {
    let reference = [1.0_f32, -2.0, 0.0, 4.0];
    let candidate = [1.1_f32, -2.2, 0.3, 3.6];
    let report = compare_f32(
        &reference,
        &candidate,
        &[2, 2],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();

    assert_eq!(report.element_count, 4);
    assert_eq!(report.finite_pair_count, 4);
    assert_close(report.max_abs, 0.4, 1.0e-6);
    assert_close(report.mean_abs, 0.25, 1.0e-6);
    // Quantiles use the deterministic nearest-rank definition.
    assert_close(report.p50_abs, 0.2, 1.0e-6);
    assert_close(report.p90_abs, 0.4, 1.0e-6);
    assert_close(report.p99_abs, 0.4, 1.0e-6);
    assert_close(report.relative_l2, (0.3_f64 / 21.0).sqrt(), 1.0e-6);
    assert_close(
        report.cosine_similarity,
        19.9 / (21.0_f64 * 19.1).sqrt(),
        1.0e-6,
    );

    let per_token = report.per_token_relative_l2.as_ref().unwrap();
    assert_eq!(per_token.len(), 2);
    assert_close(per_token[0], 0.1, 1.0e-6);
    assert_close(per_token[1], 0.125, 1.0e-6);
    let per_channel = report.per_channel_relative_l2.as_ref().unwrap();
    assert_eq!(per_channel.len(), 2);
    assert_close(per_channel[0], 0.1_f64.sqrt(), 1.0e-6);
    assert_close(per_channel[1], 0.1, 1.0e-6);

    assert_eq!(report.reference_non_finite.nan, 0);
    assert_eq!(report.reference_non_finite.positive_infinity, 0);
    assert_eq!(report.reference_non_finite.negative_infinity, 0);
    assert_eq!(report.candidate_non_finite, report.reference_non_finite);
    assert_eq!(report.non_finite_mismatches, 0);
}

#[test]
fn every_policy_metric_can_fail_independently_with_a_machine_readable_reason() {
    let report = compare_f32(
        &[1.0_f32, -2.0, 0.0, 4.0],
        &[1.1_f32, -2.2, 0.3, 3.6],
        &[2, 2],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    assert!(report.assess(&permissive_policy()).unwrap().passed());

    let cases = [
        (
            ComparisonPolicy {
                max_abs: report.max_abs.next_down(),
                ..permissive_policy()
            },
            MetricViolation::MaxAbs,
        ),
        (
            ComparisonPolicy {
                max_mean_abs: report.mean_abs.next_down(),
                ..permissive_policy()
            },
            MetricViolation::MeanAbs,
        ),
        (
            ComparisonPolicy {
                max_p99_abs: report.p99_abs.next_down(),
                ..permissive_policy()
            },
            MetricViolation::P99Abs,
        ),
        (
            ComparisonPolicy {
                max_relative_l2: report.relative_l2.next_down(),
                ..permissive_policy()
            },
            MetricViolation::RelativeL2,
        ),
        (
            ComparisonPolicy {
                min_cosine_similarity: report.cosine_similarity.next_up(),
                ..permissive_policy()
            },
            MetricViolation::CosineSimilarity,
        ),
        (
            ComparisonPolicy {
                max_per_token_relative_l2: Some(0.124),
                ..permissive_policy()
            },
            MetricViolation::PerTokenRelativeL2,
        ),
        (
            ComparisonPolicy {
                max_per_channel_relative_l2: Some(0.31),
                ..permissive_policy()
            },
            MetricViolation::PerChannelRelativeL2,
        ),
    ];

    for (policy, expected) in cases {
        let verdict = report.assess(&policy).unwrap();
        assert!(!verdict.passed());
        assert_eq!(verdict.violations(), &[expected]);
    }
}

#[test]
fn non_finite_values_are_counted_by_kind_and_cannot_accidentally_pass() {
    let report = compare_f32(
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0],
        &[
            f32::NAN,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ],
        &[4],
        ComparisonAxes::default(),
    )
    .unwrap();
    assert_eq!(report.reference_non_finite.nan, 1);
    assert_eq!(report.reference_non_finite.positive_infinity, 1);
    assert_eq!(report.reference_non_finite.negative_infinity, 1);
    assert_eq!(report.candidate_non_finite.nan, 1);
    assert_eq!(report.candidate_non_finite.positive_infinity, 1);
    assert_eq!(report.candidate_non_finite.negative_infinity, 2);
    assert_eq!(report.non_finite_mismatches, 2);
    assert_eq!(report.finite_pair_count, 0);

    let verdict = report.assess(&permissive_policy()).unwrap();
    assert_eq!(verdict.violations(), &[MetricViolation::NonFinite]);

    let allow_matching_non_finite = ComparisonPolicy {
        require_finite: false,
        ..permissive_policy()
    };
    // Mismatching infinity signs and finite/non-finite pairs remain hard failures.
    assert_eq!(
        report
            .assess(&allow_matching_non_finite)
            .unwrap()
            .violations(),
        &[MetricViolation::NonFiniteMismatch]
    );
}

#[test]
fn matching_non_finite_categories_can_pass_only_when_policy_explicitly_allows_them() {
    let report = compare_f32(
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0],
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0],
        &[4],
        ComparisonAxes::default(),
    )
    .unwrap();
    let policy = ComparisonPolicy {
        require_finite: false,
        max_abs: 0.0,
        max_mean_abs: 0.0,
        max_p99_abs: 0.0,
        max_relative_l2: 0.0,
        min_cosine_similarity: 1.0,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    };
    assert!(report.assess(&policy).unwrap().passed());
}

#[test]
fn zero_norm_and_extreme_f32_vectors_have_explicit_stable_semantics() {
    let identical_zero =
        compare_f32(&[0.0, -0.0], &[0.0, 0.0], &[2], ComparisonAxes::default()).unwrap();
    assert_eq!(identical_zero.relative_l2, 0.0);
    assert_eq!(identical_zero.cosine_similarity, 1.0);

    let zero_reference =
        compare_f32(&[0.0, 0.0], &[0.0, 1.0], &[2], ComparisonAxes::default()).unwrap();
    assert!(zero_reference.relative_l2.is_infinite());
    assert_eq!(zero_reference.cosine_similarity, 0.0);

    let extreme = compare_f32(
        &[f32::MAX, f32::MIN_POSITIVE, -f32::MAX],
        &[f32::MAX, f32::MIN_POSITIVE, -f32::MAX],
        &[3],
        ComparisonAxes::default(),
    )
    .unwrap();
    assert_eq!(extreme.max_abs, 0.0);
    assert_eq!(extreme.relative_l2, 0.0);
    assert_close(extreme.cosine_similarity, 1.0, 1.0e-15);
}

#[test]
fn malformed_shapes_axes_and_policies_are_rejected_precisely() {
    let error = compare_f32(&[], &[], &[0], ComparisonAxes::default()).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::EmptyTensor);

    let error = compare_f32(&[1.0], &[1.0, 2.0], &[1], ComparisonAxes::default()).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::LengthMismatch);

    let error = compare_f32(&[1.0; 4], &[1.0; 4], &[2, 3], ComparisonAxes::default()).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::ShapeMismatch);

    let error =
        compare_f32(&[1.0], &[1.0], &[usize::MAX, 2], ComparisonAxes::default()).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::ShapeOverflow);

    let error = compare_f32(
        &[1.0],
        &[1.0],
        &[1],
        ComparisonAxes {
            token_axis: Some(1),
            channel_axis: None,
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::AxisOutOfRange);

    let error = compare_f32(
        &[1.0],
        &[1.0],
        &[1],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(0),
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::DuplicateAxis);

    let report = compare_f32(&[1.0], &[1.0], &[1], ComparisonAxes::default()).unwrap();
    let invalid_policy = ComparisonPolicy {
        max_abs: -1.0,
        ..permissive_policy()
    };
    let error = report.assess(&invalid_policy).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::InvalidPolicy);

    let invalid_policy = ComparisonPolicy {
        min_cosine_similarity: 1.01,
        ..permissive_policy()
    };
    let error = report.assess(&invalid_policy).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::InvalidPolicy);

    for invalid in [f64::NAN, f64::NEG_INFINITY, -1.0] {
        for policy in [
            ComparisonPolicy {
                max_abs: invalid,
                ..permissive_policy()
            },
            ComparisonPolicy {
                max_mean_abs: invalid,
                ..permissive_policy()
            },
            ComparisonPolicy {
                max_p99_abs: invalid,
                ..permissive_policy()
            },
            ComparisonPolicy {
                max_relative_l2: invalid,
                ..permissive_policy()
            },
            ComparisonPolicy {
                max_per_token_relative_l2: Some(invalid),
                ..permissive_policy()
            },
            ComparisonPolicy {
                max_per_channel_relative_l2: Some(invalid),
                ..permissive_policy()
            },
        ] {
            assert_eq!(
                report.assess(&policy).unwrap_err().code(),
                ComparisonErrorCode::InvalidPolicy
            );
        }
    }
    for invalid_cosine in [f64::NAN, f64::NEG_INFINITY, -1.01, 1.01] {
        let policy = ComparisonPolicy {
            min_cosine_similarity: invalid_cosine,
            ..permissive_policy()
        };
        assert_eq!(
            report.assess(&policy).unwrap_err().code(),
            ComparisonErrorCode::InvalidPolicy
        );
    }
}

#[test]
fn ordered_errors_distinguish_nearest_rank_p50_p90_p99_and_threshold_equality_passes() {
    let reference = vec![10.0_f32; 100];
    let candidate: Vec<_> = (1..=100).map(|rank| 10.0 + rank as f32 / 100.0).collect();
    let report = compare_f32(&reference, &candidate, &[100], ComparisonAxes::default()).unwrap();
    assert_close(report.p50_abs, 0.50, 1.0e-6);
    assert_close(report.p90_abs, 0.90, 1.0e-6);
    assert_close(report.p99_abs, 0.99, 1.0e-6);
    assert_close(report.max_abs, 1.00, 1.0e-6);

    let equality = ComparisonPolicy {
        require_finite: true,
        max_abs: report.max_abs,
        max_mean_abs: report.mean_abs,
        max_p99_abs: report.p99_abs,
        max_relative_l2: report.relative_l2,
        min_cosine_similarity: report.cosine_similarity,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    };
    assert!(report.assess(&equality).unwrap().passed());

    let multiple = ComparisonPolicy {
        max_abs: 0.5,
        max_mean_abs: 0.25,
        max_p99_abs: 0.5,
        max_relative_l2: 0.01,
        min_cosine_similarity: 0.5,
        ..permissive_policy()
    };
    let violations = report.assess(&multiple).unwrap();
    assert!(violations.violations().contains(&MetricViolation::MaxAbs));
    assert!(violations.violations().contains(&MetricViolation::MeanAbs));
    assert!(violations.violations().contains(&MetricViolation::P99Abs));
    assert!(
        violations
            .violations()
            .contains(&MetricViolation::RelativeL2)
    );
    assert!(violations.violations().len() >= 4);
}

#[test]
fn quantiles_and_axis_reports_are_repeatable_and_preserve_shape_order() {
    let reference: Vec<_> = (0..24).map(|value| value as f32 + 1.0).collect();
    let candidate: Vec<_> = reference
        .iter()
        .enumerate()
        .map(|(index, value)| value + (index % 7) as f32 * 0.01)
        .collect();
    let axes = ComparisonAxes {
        token_axis: Some(1),
        channel_axis: Some(2),
    };
    let first = compare_f32(&reference, &candidate, &[2, 3, 4], axes).unwrap();
    let second = compare_f32(&reference, &candidate, &[2, 3, 4], axes).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.per_token_relative_l2.as_ref().unwrap().len(), 3);
    assert_eq!(first.per_channel_relative_l2.as_ref().unwrap().len(), 4);

    fn independent_axis_relative_l2(
        reference: &[f32],
        candidate: &[f32],
        shape: [usize; 3],
        axis: usize,
    ) -> Vec<f64> {
        let mut numerator = vec![0.0_f64; shape[axis]];
        let mut denominator = vec![0.0_f64; shape[axis]];
        for first in 0..shape[0] {
            for second in 0..shape[1] {
                for third in 0..shape[2] {
                    let coordinates = [first, second, third];
                    let index = (first * shape[1] + second) * shape[2] + third;
                    let error = candidate[index] as f64 - reference[index] as f64;
                    numerator[coordinates[axis]] += error * error;
                    denominator[coordinates[axis]] +=
                        reference[index] as f64 * reference[index] as f64;
                }
            }
        }
        numerator
            .into_iter()
            .zip(denominator)
            .map(|(numerator, denominator)| (numerator / denominator).sqrt())
            .collect()
    }

    let expected_tokens = independent_axis_relative_l2(&reference, &candidate, [2, 3, 4], 1);
    let expected_channels = independent_axis_relative_l2(&reference, &candidate, [2, 3, 4], 2);
    for (actual, expected) in first
        .per_token_relative_l2
        .as_ref()
        .unwrap()
        .iter()
        .zip(expected_tokens)
    {
        assert_close(*actual, expected, 1.0e-12);
    }
    for (actual, expected) in first
        .per_channel_relative_l2
        .as_ref()
        .unwrap()
        .iter()
        .zip(expected_channels)
    {
        assert_close(*actual, expected, 1.0e-12);
    }
}

#[test]
fn logits_report_covers_topology_distribution_metrics_and_stable_token_rule() {
    fn softmax_selected(logits: &[f32], indices: &[usize]) -> Vec<f64> {
        let maximum = indices
            .iter()
            .map(|&index| logits[index] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut values: Vec<_> = indices
            .iter()
            .map(|&index| (logits[index] as f64 - maximum).exp())
            .collect();
        let denominator: f64 = values.iter().sum();
        for value in &mut values {
            *value /= denominator;
        }
        values
    }

    let reference = [4.0_f32, 3.0, 1.0, 0.0];
    let candidate = [3.8_f32, 0.9, 3.2, -0.1];
    let report = compare_logits(&reference, &candidate, 2).unwrap();
    assert_eq!(report.reference_top1, 0);
    assert_eq!(report.candidate_top1, 0);
    assert!(report.top1_agreement);
    assert_eq!(report.reference_top_k, vec![0, 1]);
    assert_eq!(report.candidate_top_k, vec![0, 2]);
    assert_eq!(report.top_k_overlap, 1);
    assert_eq!(report.top_k_overlap_fraction, 0.5);
    assert_eq!(report.reference_margin, 1.0);
    assert_eq!(report.selected_indices, vec![0, 1, 2]);
    assert_eq!(
        report
            .top_candidate_errors
            .iter()
            .map(|entry| entry.index)
            .collect::<Vec<_>>(),
        report.selected_indices
    );
    for entry in &report.top_candidate_errors {
        let expected_reference = reference[entry.index];
        let expected_candidate = candidate[entry.index];
        assert_eq!(entry.reference_logit, expected_reference);
        assert_eq!(entry.candidate_logit, expected_candidate);
        assert_close(
            entry.absolute_error,
            (expected_reference as f64 - expected_candidate as f64).abs(),
            0.0,
        );
    }

    let reference_distribution = softmax_selected(&reference, &report.selected_indices);
    let candidate_distribution = softmax_selected(&candidate, &report.selected_indices);
    let kl: f64 = reference_distribution
        .iter()
        .zip(&candidate_distribution)
        .map(|(reference, candidate)| reference * (reference / candidate).ln())
        .sum();
    let midpoint: Vec<_> = reference_distribution
        .iter()
        .zip(&candidate_distribution)
        .map(|(reference, candidate)| (reference + candidate) * 0.5)
        .collect();
    let js = 0.5
        * reference_distribution
            .iter()
            .zip(&midpoint)
            .map(|(reference, midpoint)| reference * (reference / midpoint).ln())
            .sum::<f64>()
        + 0.5
            * candidate_distribution
                .iter()
                .zip(&midpoint)
                .map(|(candidate, midpoint)| candidate * (candidate / midpoint).ln())
                .sum::<f64>();
    assert_close(report.kl_reference_to_candidate, kl, 1.0e-12);
    assert_close(report.jensen_shannon_divergence, js, 1.0e-12);
    assert_eq!(
        report.stable_token_verdict(0.4).unwrap(),
        StableTokenVerdict::RequiredAndMatched
    );

    let tied = compare_logits(&[2.0, 2.0, 1.0], &[1.9, 2.1, 1.0], 2).unwrap();
    assert_eq!(tied.reference_top1, 0, "ties choose the smaller token id");
    assert_eq!(tied.candidate_top1, 1);
    assert_eq!(tied.reference_margin, 0.0);
    assert_eq!(
        tied.stable_token_verdict(0.01).unwrap(),
        StableTokenVerdict::Ambiguous
    );

    let changed = compare_logits(&[5.0, 1.0, 0.0], &[0.0, 6.0, 0.0], 2).unwrap();
    assert_eq!(
        changed.stable_token_verdict(1.0).unwrap(),
        StableTokenVerdict::RequiredButChanged
    );
}

#[test]
fn logits_contract_rejects_bad_k_lengths_nonfinite_values_and_error_envelopes() {
    for k in [0, 4] {
        let error = compare_logits(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], k).unwrap_err();
        assert_eq!(error.code(), ComparisonErrorCode::InvalidTopK);
    }
    let error = compare_logits(&[1.0], &[1.0, 2.0], 1).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::LengthMismatch);
    let error = compare_logits(&[f32::NAN], &[1.0], 1).unwrap_err();
    assert_eq!(error.code(), ComparisonErrorCode::NonFiniteInput);
    for candidate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = compare_logits(&[1.0], &[candidate], 1).unwrap_err();
        assert_eq!(error.code(), ComparisonErrorCode::NonFiniteInput);
    }

    let report = compare_logits(&[2.0, 1.0], &[2.0, 1.0], 1).unwrap();
    for envelope in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            report.stable_token_verdict(envelope).unwrap_err().code(),
            ComparisonErrorCode::InvalidPolicy
        );
    }
}
