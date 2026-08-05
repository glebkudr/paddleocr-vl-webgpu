use pvlc_cpu_ref::{
    CpuRefError, KvBlockOrder, add_vectors_f32, apply_rope_neox, gelu_erf_f32, gelu_pytorch_tanh,
    gemm_f32, layer_norm_f32, patch_projection_f32, rms_norm_f32, silu,
    streaming_segmented_attention_f32,
};
use pvlc_runtime_core::{InvocationError, KernelId, KernelInvocation};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ComparisonPolicy;

pub const M2_BOUNDARIES: [u32; 25] = [
    1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
    1023, 1024,
];
pub const M2_INPUT_FAMILIES: [&str; 8] = [
    "zeros",
    "ones",
    "alternating-signs",
    "tiny",
    "near-fp16-limit",
    "impulse",
    "repeated-pattern",
    "random",
];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M2CasePolicy {
    pub max_abs: f64,
    pub max_mean_abs: f64,
    pub max_p99_abs: f64,
    pub max_relative_l2: f64,
    pub min_cosine_similarity: f64,
    pub native_max_abs: f64,
    pub native_max_relative_l2: f64,
}

impl M2CasePolicy {
    #[must_use]
    pub const fn comparison_policy(self) -> ComparisonPolicy {
        ComparisonPolicy {
            require_finite: true,
            max_abs: self.max_abs,
            max_mean_abs: self.max_mean_abs,
            max_p99_abs: self.max_p99_abs,
            max_relative_l2: self.max_relative_l2,
            min_cosine_similarity: self.min_cosine_similarity,
            max_per_token_relative_l2: None,
            max_per_channel_relative_l2: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M2PrimitiveCase {
    pub id: String,
    pub kernel: KernelId,
    pub tags: Vec<String>,
    pub invocation: KernelInvocation,
    pub expected: Vec<f32>,
    pub shape: Vec<usize>,
    pub policy: M2CasePolicy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M2PrimitiveCorpus {
    pub schema_version: u32,
    pub oracle: String,
    pub cases: Vec<M2PrimitiveCase>,
}

#[derive(Debug, Error)]
pub enum M2CorpusError {
    #[error("M2 corpus invocation is invalid: {0}")]
    Invocation(#[from] InvocationError),
    #[error("M2 CPU oracle failed: {0}")]
    Cpu(#[from] CpuRefError),
}

pub fn m2_primitive_corpus() -> Result<M2PrimitiveCorpus, M2CorpusError> {
    let mut cases = Vec::with_capacity(400);
    add_boundary_cases(&mut cases)?;
    add_family_cases(&mut cases)?;
    add_rope_variant_cases(&mut cases)?;
    Ok(M2PrimitiveCorpus {
        schema_version: 1,
        oracle: "pvlc-cpu-ref/f32-v1".to_owned(),
        cases,
    })
}

fn add_boundary_cases(cases: &mut Vec<M2PrimitiveCase>) -> Result<(), M2CorpusError> {
    let mut seed = 1_u32;
    for (axis, axis_name) in ["rows", "inner", "columns"].into_iter().enumerate() {
        for boundary in M2_BOUNDARIES {
            let (rows, inner, columns) = match axis {
                0 => (boundary, 7, 9),
                1 => (7, boundary, 9),
                2 => (7, 9, boundary),
                _ => unreachable!(),
            };
            push_case(
                cases,
                format!("gemm_f32/boundary-{axis_name}-{boundary:04}"),
                ["boundary".to_owned(), format!("axis:{axis_name}")],
                KernelInvocation::GemmF32 {
                    rows,
                    inner,
                    columns,
                    left: deterministic_values((rows * inner) as usize, seed),
                    right: deterministic_values((inner * columns) as usize, seed + 10),
                },
                policy(KernelId::GemmF32, false),
            )?;
            seed += 1;
        }
    }

    for boundary in M2_BOUNDARIES {
        let rows = boundary;
        let columns = [1, 3, 17, 33][seed as usize % 4];
        push_case(
            cases,
            format!("gemv_f32/boundary-rows-{boundary:04}"),
            ["axis:rows".to_owned(), "boundary".to_owned()],
            KernelInvocation::GemvF32 {
                rows,
                columns,
                matrix: deterministic_values((rows * columns) as usize, seed),
                vector: deterministic_values(columns as usize, seed + 10),
            },
            policy(KernelId::GemvF32, false),
        )?;
        seed += 1;
    }
    for boundary in M2_BOUNDARIES {
        let rows = 7;
        let columns = boundary;
        push_case(
            cases,
            format!("gemv_f32/boundary-columns-{boundary:04}"),
            ["axis:columns".to_owned(), "boundary".to_owned()],
            KernelInvocation::GemvF32 {
                rows,
                columns,
                matrix: deterministic_values((rows * columns) as usize, seed),
                vector: deterministic_values(columns as usize, seed + 10),
            },
            policy(KernelId::GemvF32, false),
        )?;
        seed += 1;
    }

    for kernel in [KernelId::LayerNormF32, KernelId::RmsNormF32] {
        for boundary in M2_BOUNDARIES {
            let rows = 3;
            let width = boundary;
            push_norm_case(cases, kernel, rows, width, "width", boundary, seed)?;
            seed += 1;
        }
        for boundary in M2_BOUNDARIES {
            let rows = boundary;
            let width = 17;
            push_norm_case(cases, kernel, rows, width, "rows", boundary, seed)?;
            seed += 1;
        }
    }

    for boundary in M2_BOUNDARIES {
        let mut values = deterministic_values(boundary as usize, seed);
        if values.len() >= 7 {
            values[..7].copy_from_slice(&[-65_504.0, -20.0, -1.0e-7, 0.0, 1.0e-7, 20.0, 65_504.0]);
        }
        for kernel in [KernelId::SiluF32, KernelId::GeluTanhF32] {
            let invocation = match kernel {
                KernelId::SiluF32 => KernelInvocation::SiluF32 {
                    values: values.clone(),
                },
                KernelId::GeluTanhF32 => KernelInvocation::GeluTanhF32 {
                    values: values.clone(),
                },
                _ => unreachable!(),
            };
            push_case(
                cases,
                format!("{kernel}/boundary-length-{boundary:04}"),
                ["axis:length".to_owned(), "boundary".to_owned()],
                invocation,
                policy(kernel, false),
            )?;
        }
        seed += 1;
    }

    for boundary in M2_BOUNDARIES {
        let rows = boundary;
        let width = 5;
        let rotary_dim = 2;
        push_case(
            cases,
            format!("rope_neox_f32/boundary-rows-{boundary:04}"),
            ["axis:rows".to_owned(), "boundary".to_owned()],
            KernelInvocation::RopeNeoxF32 {
                rows,
                width,
                rotary_dim,
                positions: (0..rows).map(|row| row * 7).collect(),
                base: [2.0, 10_000.0, 500_000.0][seed as usize % 3],
                values: deterministic_values((rows * width) as usize, seed),
            },
            policy(KernelId::RopeNeoxF32, false),
        )?;
        seed += 1;
    }
    Ok(())
}

fn push_norm_case(
    cases: &mut Vec<M2PrimitiveCase>,
    kernel: KernelId,
    rows: u32,
    width: u32,
    axis_name: &str,
    boundary: u32,
    seed: u32,
) -> Result<(), M2CorpusError> {
    let input = deterministic_values((rows * width) as usize, seed);
    let weight = deterministic_values(width as usize, seed + 10)
        .into_iter()
        .map(|value| 1.0 + value * 0.1)
        .collect::<Vec<_>>();
    let epsilon = [1.0e-6, 1.0e-5, 1.0e-3, 0.5][seed as usize % 4];
    let invocation = match kernel {
        KernelId::LayerNormF32 => KernelInvocation::LayerNormF32 {
            rows,
            width,
            input,
            weight,
            bias: deterministic_values(width as usize, seed + 20)
                .into_iter()
                .map(|value| value * 0.05)
                .collect(),
            epsilon,
        },
        KernelId::RmsNormF32 => KernelInvocation::RmsNormF32 {
            rows,
            width,
            input,
            weight,
            epsilon,
        },
        _ => unreachable!(),
    };
    push_case(
        cases,
        format!("{kernel}/boundary-{axis_name}-{boundary:04}"),
        ["boundary".to_owned(), format!("axis:{axis_name}")],
        invocation,
        policy(kernel, false),
    )
}

fn add_family_cases(cases: &mut Vec<M2PrimitiveCase>) -> Result<(), M2CorpusError> {
    let mut seed = 1_001_u32;
    for family in M2_INPUT_FAMILIES {
        for operand in ["left", "right"] {
            let rows = 4;
            let inner = 17;
            let columns = 4;
            let mut left = deterministic_values((rows * inner) as usize, seed);
            let mut right = deterministic_values((inner * columns) as usize, seed + 1);
            *match operand {
                "left" => &mut left,
                "right" => &mut right,
                _ => unreachable!(),
            } = family_values(
                family,
                if operand == "left" {
                    left.len()
                } else {
                    right.len()
                },
                seed,
            );
            push_family_case(
                cases,
                family,
                operand,
                KernelInvocation::GemmF32 {
                    rows,
                    inner,
                    columns,
                    left,
                    right,
                },
            )?;
            seed += 1;
        }

        for operand in ["matrix", "vector"] {
            let rows = 8;
            let columns = 17;
            let mut matrix = deterministic_values((rows * columns) as usize, seed);
            let mut vector = deterministic_values(columns as usize, seed + 1);
            *match operand {
                "matrix" => &mut matrix,
                "vector" => &mut vector,
                _ => unreachable!(),
            } = family_values(
                family,
                if operand == "matrix" {
                    matrix.len()
                } else {
                    vector.len()
                },
                seed,
            );
            push_family_case(
                cases,
                family,
                operand,
                KernelInvocation::GemvF32 {
                    rows,
                    columns,
                    matrix,
                    vector,
                },
            )?;
            seed += 1;
        }

        let rows = 2;
        let width = 17;
        let input = family_values(family, (rows * width) as usize, seed);
        let weight = deterministic_values(width as usize, seed + 1)
            .into_iter()
            .map(|value| 1.0 + value * 0.1)
            .collect::<Vec<_>>();
        push_family_case(
            cases,
            family,
            "input",
            KernelInvocation::LayerNormF32 {
                rows,
                width,
                input: input.clone(),
                weight: weight.clone(),
                bias: vec![0.125; width as usize],
                epsilon: 1.0e-5,
            },
        )?;
        push_family_case(
            cases,
            family,
            "input",
            KernelInvocation::RmsNormF32 {
                rows,
                width,
                input,
                weight,
                epsilon: 1.0e-5,
            },
        )?;

        let values = family_values(family, 65, seed + 2);
        push_family_case(
            cases,
            family,
            "values",
            KernelInvocation::SiluF32 {
                values: values.clone(),
            },
        )?;
        push_family_case(
            cases,
            family,
            "values",
            KernelInvocation::GeluTanhF32 { values },
        )?;

        push_family_case(
            cases,
            family,
            "values",
            KernelInvocation::RopeNeoxF32 {
                rows: 4,
                width: 10,
                rotary_dim: 8,
                positions: vec![0, 1, 17, 127],
                base: [2.0, 10_000.0, 500_000.0][seed as usize % 3],
                values: family_values(family, 40, seed + 3),
            },
        )?;
        seed += 4;
    }
    Ok(())
}

fn push_family_case(
    cases: &mut Vec<M2PrimitiveCase>,
    family: &str,
    operand: &str,
    invocation: KernelInvocation,
) -> Result<(), M2CorpusError> {
    let kernel = invocation.kernel_id();
    push_case(
        cases,
        format!("{kernel}/family-{family}-{operand}"),
        [
            format!("family-operand:{operand}"),
            format!("family:{family}"),
        ],
        invocation,
        policy(kernel, true),
    )
}

fn add_rope_variant_cases(cases: &mut Vec<M2PrimitiveCase>) -> Result<(), M2CorpusError> {
    for (index, rotary_dim) in [2, 4, 8, 16, 32, 64].into_iter().enumerate() {
        let rows = 4;
        let width = rotary_dim + 3;
        push_case(
            cases,
            format!("rope_neox_f32/rotary-dim-{rotary_dim:02}"),
            ["rope-variant".to_owned()],
            KernelInvocation::RopeNeoxF32 {
                rows,
                width,
                rotary_dim,
                positions: vec![0, 1, 17, 4096],
                base: [2.0, 10_000.0, 500_000.0, 1_000_000.0][index % 4],
                values: deterministic_values((rows * width) as usize, 2_001 + index as u32),
            },
            policy(KernelId::RopeNeoxF32, false),
        )?;
    }
    Ok(())
}

fn push_case<const N: usize>(
    cases: &mut Vec<M2PrimitiveCase>,
    id: String,
    tags: [String; N],
    invocation: KernelInvocation,
    policy: M2CasePolicy,
) -> Result<(), M2CorpusError> {
    let plan = invocation.plan()?;
    let shape = output_shape(&invocation);
    let expected = cpu_expected(&invocation)?;
    debug_assert_eq!(expected.len(), plan.output_elements);
    let mut tags = tags.into_iter().collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    cases.push(M2PrimitiveCase {
        id,
        kernel: invocation.kernel_id(),
        tags,
        invocation,
        expected,
        shape,
        policy,
    });
    Ok(())
}

fn cpu_expected(invocation: &KernelInvocation) -> Result<Vec<f32>, CpuRefError> {
    match invocation {
        KernelInvocation::GemmF32 {
            rows,
            inner,
            columns,
            left,
            right,
        } => gemm_f32(
            left,
            *rows as usize,
            *inner as usize,
            right,
            *columns as usize,
        ),
        KernelInvocation::GemvF32 {
            rows,
            columns,
            matrix,
            vector,
        } => gemm_f32(matrix, *rows as usize, *columns as usize, vector, 1),
        KernelInvocation::LayerNormF32 {
            rows,
            width,
            input,
            weight,
            bias,
            epsilon,
        } => layer_norm_f32(
            input,
            *rows as usize,
            *width as usize,
            weight,
            bias,
            *epsilon,
        ),
        KernelInvocation::RmsNormF32 {
            rows,
            width,
            input,
            weight,
            epsilon,
        } => rms_norm_f32(input, *rows as usize, *width as usize, weight, *epsilon),
        KernelInvocation::SiluF32 { values } => Ok(values.iter().copied().map(silu).collect()),
        KernelInvocation::GeluTanhF32 { values } => {
            Ok(values.iter().copied().map(gelu_pytorch_tanh).collect())
        }
        KernelInvocation::GeluErfF32 { values } => {
            Ok(values.iter().copied().map(gelu_erf_f32).collect())
        }
        KernelInvocation::RopeNeoxF32 {
            rows,
            width,
            rotary_dim,
            positions,
            base,
            values,
        } => {
            let mut output = values.clone();
            apply_rope_neox(
                &mut output,
                *rows as usize,
                *width as usize,
                *rotary_dim as usize,
                positions,
                *base,
            )?;
            Ok(output)
        }
        KernelInvocation::VisionAttentionF32 {
            tokens,
            heads,
            head_dim,
            query,
            key,
            value,
            cu_seqlens,
        } => {
            let boundaries = cu_seqlens
                .iter()
                .map(|boundary| *boundary as usize)
                .collect::<Vec<_>>();
            streaming_segmented_attention_f32(
                query,
                key,
                value,
                *tokens as usize,
                *heads as usize,
                *head_dim as usize,
                &boundaries,
                17,
                KvBlockOrder::Forward,
            )
        }
        KernelInvocation::VisionPatchProjectionF32 {
            patch_count,
            input_width,
            output_width,
            input,
            weight,
            bias,
        } => patch_projection_f32(
            input,
            *patch_count as usize,
            *input_width as usize,
            1,
            weight,
            bias,
            *output_width as usize,
        ),
        KernelInvocation::AddF32 { left, right } => add_vectors_f32(left, right),
        KernelInvocation::ProjectorMerge2x2F32 {
            output_tokens,
            hidden_size,
            input,
            source_token_indices,
        } => {
            let output_tokens = *output_tokens as usize;
            let hidden_size = *hidden_size as usize;
            let merged_width = hidden_size * 4;
            let mut output = vec![0.0; output_tokens * merged_width];
            for output_token in 0..output_tokens {
                for source_patch in 0..4 {
                    let source_token =
                        source_token_indices[output_token * 4 + source_patch] as usize;
                    let source_start = source_token * hidden_size;
                    let target_start = output_token * merged_width + source_patch * hidden_size;
                    output[target_start..target_start + hidden_size]
                        .copy_from_slice(&input[source_start..source_start + hidden_size]);
                }
            }
            Ok(output)
        }
    }
}

fn output_shape(invocation: &KernelInvocation) -> Vec<usize> {
    match invocation {
        KernelInvocation::GemmF32 { rows, columns, .. } => {
            vec![*rows as usize, *columns as usize]
        }
        KernelInvocation::GemvF32 { rows, .. } => vec![*rows as usize],
        KernelInvocation::LayerNormF32 { rows, width, .. }
        | KernelInvocation::RmsNormF32 { rows, width, .. }
        | KernelInvocation::RopeNeoxF32 { rows, width, .. } => {
            vec![*rows as usize, *width as usize]
        }
        KernelInvocation::SiluF32 { values }
        | KernelInvocation::GeluTanhF32 { values }
        | KernelInvocation::GeluErfF32 { values } => {
            vec![values.len()]
        }
        KernelInvocation::VisionAttentionF32 {
            tokens,
            heads,
            head_dim,
            ..
        } => vec![*tokens as usize, *heads as usize, *head_dim as usize],
        KernelInvocation::VisionPatchProjectionF32 {
            patch_count,
            output_width,
            ..
        } => vec![*patch_count as usize, *output_width as usize],
        KernelInvocation::AddF32 { left, .. } => vec![left.len()],
        KernelInvocation::ProjectorMerge2x2F32 {
            output_tokens,
            hidden_size,
            ..
        } => vec![*output_tokens as usize, *hidden_size as usize * 4],
    }
}

fn family_values(family: &str, length: usize, seed: u32) -> Vec<f32> {
    match family {
        "zeros" => vec![0.0; length],
        "ones" => vec![1.0; length],
        "alternating-signs" => (0..length)
            .map(|index| if index.is_multiple_of(2) { -1.0 } else { 1.0 })
            .collect(),
        "tiny" => vec![1.0e-30; length],
        "near-fp16-limit" => (0..length)
            .map(|index| {
                if index.is_multiple_of(2) {
                    -65_504.0
                } else {
                    65_504.0
                }
            })
            .collect(),
        "impulse" => {
            let mut values = vec![0.0; length];
            values[length / 2] = 1.0;
            values
        }
        "repeated-pattern" => (0..length)
            .map(|index| [0.25, -0.5, 1.0][index % 3])
            .collect(),
        "random" => deterministic_values(length, seed),
        _ => unreachable!("M2 family catalog is closed"),
    }
}

fn deterministic_values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let value = ((index as u32 * 37 + seed * 17) % 101) as i32 - 50;
            value as f32 / 32.0
        })
        .collect()
}

const fn policy(kernel: KernelId, family_case: bool) -> M2CasePolicy {
    let (max_abs, max_relative_l2) = match (kernel, family_case) {
        (KernelId::GemmF32, true) => (0.1, 3.0e-5),
        (KernelId::GemvF32, true) => (1.0, 5.0e-5),
        (KernelId::GemmF32 | KernelId::GemvF32, false) => (8.0e-5, 3.0e-5),
        (KernelId::GemvTiledF32, _) => {
            panic!("gemv_tiled_f32 is outside the accepted M2 conformance corpus")
        }
        (KernelId::LayerNormF32 | KernelId::RmsNormF32, _) => (2.0e-4, 1.0e-4),
        (KernelId::SiluF32 | KernelId::GeluTanhF32, true) => (5.0e-2, 5.0e-5),
        (KernelId::SiluF32 | KernelId::GeluTanhF32, false) => (3.0e-5, 2.0e-5),
        (KernelId::RopeNeoxF32, true) => (1.0, 5.0e-5),
        (KernelId::RopeNeoxF32, false) => (3.0e-4, 2.0e-4),
        (KernelId::VisionAttentionF32, _) => (3.0e-4, 1.0e-4),
        (KernelId::VisionPatchProjectionF32, _) => (2.0e-4, 6.0e-5),
        (KernelId::AddF32, _) => (1.0e-7, 1.0e-7),
        (KernelId::GeluErfF32, _) => (3.0e-6, 2.0e-6),
        (KernelId::ProjectorMerge2x2F32, _) => (1.0e-7, 1.0e-7),
        (
            KernelId::VisionQkvFusedF32
            | KernelId::VisionRope2dF32
            | KernelId::DecoderKvAppendF32
            | KernelId::DecoderGqaF32
            | KernelId::DecoderGqaSplitPartialF32
            | KernelId::DecoderGqaSplitMergeF32
            | KernelId::DecoderMropeF32
            | KernelId::DecoderSwigluF32
            | KernelId::DecoderPrefillGqaF32
            | KernelId::DecoderPrefillMropeF32
            | KernelId::DecoderKvAppendRangeF32
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
            | KernelId::GeluErfF16,
            _,
        ) => panic!("post-M2 kernel cannot receive an M2 corpus policy"),
    };
    let native_max_abs = match kernel {
        KernelId::GemvF32 if max_abs > 0.25 => 0.25,
        KernelId::SiluF32 | KernelId::GeluTanhF32 | KernelId::GeluErfF32 if max_abs > 1.0e-2 => {
            1.0e-2
        }
        KernelId::RopeNeoxF32 if !family_case => 2.0e-4,
        KernelId::RopeNeoxF32 if max_abs > 0.1 => 0.1,
        _ => max_abs * 0.5,
    };
    M2CasePolicy {
        max_abs,
        max_mean_abs: max_abs,
        max_p99_abs: max_abs,
        max_relative_l2,
        min_cosine_similarity: 0.999_99,
        native_max_abs,
        native_max_relative_l2: max_relative_l2 * 0.5,
    }
}
