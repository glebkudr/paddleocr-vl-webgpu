use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, module, validate_source_contract};

const FULL_FP16_KERNELS: [KernelId; 6] = [
    KernelId::LayerNormF16,
    KernelId::LinearProjectionF16,
    KernelId::VisionAttentionF16,
    KernelId::AddF16,
    KernelId::GeluTanhF16,
    KernelId::VisionRope2dF16,
];

fn source(kernel: KernelId) -> &'static str {
    module(kernel).unwrap().source
}

#[test]
fn every_full_fp16_kernel_is_catalogued_validated_and_feature_gated() {
    for kernel in FULL_FP16_KERNELS {
        let module = module(kernel).expect("full FP16 kernel must be in the production catalog");
        assert_eq!(module.spec.required_features, ["shader_f16"]);
        validate_source_contract(&module.spec, module.source).unwrap();
    }
}

#[test]
fn full_fp16_abis_never_widen_activation_or_vector_bindings_to_f32() {
    for kernel in [
        KernelId::LayerNormF16,
        KernelId::LinearProjectionF16,
        KernelId::VisionAttentionF16,
        KernelId::AddF16,
        KernelId::GeluTanhF16,
    ] {
        let module = module(kernel).unwrap();
        assert!(module.spec.bindings.iter().any(|binding| matches!(
            binding.kind,
            BindingKind::StorageReadVec4F16 | BindingKind::StorageReadWriteVec4F16
        )),);
        assert!(
            module.spec.bindings.iter().all(|binding| !matches!(
                binding.kind,
                BindingKind::StorageReadF32
                    | BindingKind::StorageReadVec4F32
                    | BindingKind::StorageReadWriteF32
            )),
            "{kernel} must not expose an F32 activation/vector storage binding",
        );
    }

    let rope = module(KernelId::VisionRope2dF16).unwrap();
    assert_eq!(
        rope.spec
            .bindings
            .iter()
            .map(|binding| binding.kind)
            .collect::<Vec<_>>(),
        vec![
            BindingKind::StorageReadWriteF16,
            BindingKind::StorageReadWriteF16,
            BindingKind::StorageReadF32,
            BindingKind::StorageReadF32,
            BindingKind::Uniform,
        ],
        "only the precomputed trigonometric tables remain F32",
    );
}

#[test]
fn projection_uses_half_storage_with_f32_accumulation_and_half_output() {
    let source = source(KernelId::LinearProjectionF16);
    for required in [
        "var<workgroup> input_tile: array<array<vec4<f16>",
        "var<workgroup> weight_tile: array<array<vec4<f16>",
        "var accumulators: array<vec4<f32>",
        "fma(vec4<f32>(f32(activation.x)), vec4<f32>(coefficient0)",
        "let bias_value = vec4<f32>(bias.data[global_column_vec])",
        "output.data[output_row * output_width_vec + global_column_vec] =",
        "vec4<f16>(values)",
    ] {
        assert!(
            source.contains(required),
            "full FP16 projection is missing {required:?}",
        );
    }
    for forbidden in [
        "data: array<vec4<f32>>",
        "output.data[output_base + 0u]",
    ] {
        assert!(
            !source.contains(forbidden),
            "full FP16 projection still widens through {forbidden:?}",
        );
    }
}

#[test]
fn normalization_and_attention_keep_stability_reductions_in_f32_only() {
    let norm = source(KernelId::LayerNormF16);
    assert!(norm.contains("var mean = 0.0f"));
    assert!(norm.contains("var variance = 0.0f"));
    assert!(norm.contains("vec4<f16>("));
    assert!(!norm.contains("var<storage, read> input: F32"));

    let attention = source(KernelId::VisionAttentionF16);
    assert!(attention.contains("array<vec4<f16>, 18>"));
    assert!(attention.contains("vec4<f32>(query_vectors[vector_index])"));
    assert!(attention.contains(
        "vec4<f32>(key_cache[key_slot * MAX_HEAD_VECTORS + vector_index])",
    ));
    assert!(attention.contains("var running_denominator = 0.0f"));
    assert!(attention.contains("var scores: array<f32, 16>"));
    assert!(attention.contains("output.data["));
    assert!(attention.contains("vec4<f16>(normalized)"));
}

#[test]
fn elementwise_and_rope_kernels_write_half_storage_directly() {
    let add = source(KernelId::AddF16);
    assert!(add.contains("output.data[index] = left.data[index] + right.data[index]"));
    assert!(add.contains("array<vec4<f16>>"));

    let gelu = source(KernelId::GeluTanhF16);
    assert!(gelu.contains("let value = input.data[index]"));
    assert!(gelu.contains("let cubic = value * value * value"));
    assert!(gelu.contains("vec4<f16>(0.7978846h)"));
    assert!(!gelu.contains("vec4<f32>"));
    assert!(!gelu.contains("f32("));

    let rope = source(KernelId::VisionRope2dF16);
    assert!(rope.contains("array<f16>"));
    assert!(rope.contains("f16(query_first * cosine - query_second * sine)"));
    assert!(rope.contains("f16(key_second * cosine + key_first * sine)"));
}

#[test]
fn semantic_mutation_audit_rejects_disconnected_or_widened_full_fp16_dataflow() {
    for kernel in FULL_FP16_KERNELS {
        audit_full_fp16_kernel(kernel, source(kernel)).unwrap();
    }

    let mutations = [
        (
            KernelId::LayerNormF16,
            "mean = mean + f32(value.x) + f32(value.y) + f32(value.z) + f32(value.w);",
            "mean = mean + f32(value.x);",
        ),
        (
            KernelId::LayerNormF16,
            "variance = variance + dot(centered, centered);",
            "variance = variance + centered.x * centered.x;",
        ),
        (
            KernelId::LayerNormF16,
            "output.data[row_start_vec + column_vec] = vec4<f16>(normalized * scale + shift);",
            "output.data[row_start_vec + column_vec] = weight.data[column_vec];",
        ),
        (
            KernelId::LinearProjectionF16,
            "input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];",
            "input_tile[tile_row][local_x] = vec4<f16>(0.0h);",
        ),
        (
            KernelId::LinearProjectionF16,
            "weight_tile[tile_depth][local_x] = weight.data[input_depth * output_width_vec + global_column_vec];",
            "weight_tile[tile_depth][local_x] = vec4<f16>(0.0h);",
        ),
        (
            KernelId::LinearProjectionF16,
            "accumulators[row_offset] = fma(vec4<f32>(f32(activation.w)), vec4<f32>(coefficient3), accumulators[row_offset]);",
            "accumulators[row_offset] = accumulators[row_offset];",
        ),
        (
            KernelId::LinearProjectionF16,
            "let values = accumulators[row_offset] + bias_value;",
            "let values = bias_value;",
        ),
        (
            KernelId::VisionAttentionF16,
            "query_vectors[vector_index] = query.data[query_source];",
            "query_vectors[vector_index] = vec4<f16>(0.0h);",
        ),
        (
            KernelId::VisionAttentionF16,
            "scores[key_slot] = scores[key_slot] + dot(\n                        vec4<f32>(query_vectors[vector_index]),\n                        vec4<f32>(key_cache[key_slot * MAX_HEAD_VECTORS + vector_index]),\n                    );",
            "scores[key_slot] = scores[key_slot];",
        ),
        (
            KernelId::VisionAttentionF16,
            "block_weighted = block_weighted + scores[key_slot] * vec4<f32>(value_cache[key_slot * MAX_HEAD_VECTORS + vector_index]);",
            "block_weighted = block_weighted;",
        ),
        (
            KernelId::VisionAttentionF16,
            "let normalized = attention_output[vector_index] / running_denominator;",
            "let normalized = attention_output[vector_index];",
        ),
        (
            KernelId::AddF16,
            "output.data[index] = left.data[index] + right.data[index];",
            "output.data[index] = left.data[index];",
        ),
        (
            KernelId::GeluTanhF16,
            "let argument = vec4<f16>(0.7978846h) * (value + vec4<f16>(0.044715h) * cubic);",
            "let argument = value;",
        ),
        (
            KernelId::GeluTanhF16,
            "output.data[index] = vec4<f16>(0.5h) * value * (vec4<f16>(1.0h) + tanh(argument));",
            "output.data[index] = value;",
        ),
        (
            KernelId::VisionRope2dF16,
            "let first_index = (token * params.heads + head) * params.head_dim + pair;",
            "let first_index = pair;",
        ),
        (
            KernelId::VisionRope2dF16,
            "query.data[first_index] = f16(query_first * cosine - query_second * sine);",
            "query.data[first_index] = f16(query_first);",
        ),
        (
            KernelId::VisionRope2dF16,
            "key.data[second_index] = f16(key_second * cosine + key_first * sine);",
            "key.data[second_index] = f16(key_second);",
        ),
    ];
    for (kernel, from, to) in mutations {
        let production = source(kernel);
        assert!(
            production.contains(from),
            "mutation fixture drifted from {kernel}: {from:?}",
        );
        let mutated = production.replace(from, to);
        assert!(
            audit_full_fp16_kernel(kernel, &mutated).is_err(),
            "semantic audit accepted {kernel} mutation {from:?} -> {to:?}",
        );
    }
}

fn audit_full_fp16_kernel(kernel: KernelId, source: &str) -> Result<(), String> {
    let required: &[&str] = match kernel {
        KernelId::LayerNormF16 => &[
            "let value = input.data[row_start_vec + column_vec];",
            "mean = mean + f32(value.x) + f32(value.y) + f32(value.z) + f32(value.w);",
            "mean = mean / f32(params.width);",
            "variance = variance + dot(centered, centered);",
            "let inverse_stddev = inverseSqrt(variance / f32(params.width) + params.epsilon);",
            "let scale = vec4<f32>(weight.data[column_vec]);",
            "let shift = vec4<f32>(bias.data[column_vec]);",
            "output.data[row_start_vec + column_vec] = vec4<f16>(normalized * scale + shift);",
        ],
        KernelId::LinearProjectionF16 => &[
            "let input_width_vec = params.input_width / 4u;",
            "let output_width_vec = params.output_width / 4u;",
            "input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];",
            "weight_tile[tile_depth][local_x] = weight.data[input_depth * output_width_vec + global_column_vec];",
            "let activation = input_tile[local_y * 4u + row_offset][depth_vector];",
            "accumulators[row_offset] = fma(vec4<f32>(f32(activation.x)), vec4<f32>(coefficient0), accumulators[row_offset]);",
            "accumulators[row_offset] = fma(vec4<f32>(f32(activation.y)), vec4<f32>(coefficient1), accumulators[row_offset]);",
            "accumulators[row_offset] = fma(vec4<f32>(f32(activation.z)), vec4<f32>(coefficient2), accumulators[row_offset]);",
            "accumulators[row_offset] = fma(vec4<f32>(f32(activation.w)), vec4<f32>(coefficient3), accumulators[row_offset]);",
            "let bias_value = vec4<f32>(bias.data[global_column_vec]);",
            "let values = accumulators[row_offset] + bias_value;",
            "output.data[output_row * output_width_vec + global_column_vec] = vec4<f16>(values);",
        ],
        KernelId::VisionAttentionF16 => &[
            "query_vectors[vector_index] = query.data[query_source];",
            "key_cache[cache_index] = loaded_key;",
            "value_cache[cache_index] = loaded_value;",
            "scores[key_slot] = scores[key_slot] + dot(\n                        vec4<f32>(query_vectors[vector_index]),\n                        vec4<f32>(key_cache[key_slot * MAX_HEAD_VECTORS + vector_index]),\n                    );",
            "scores[key_slot] = scores[key_slot] * attention_scale;",
            "scores[key_slot] = exp(scores[key_slot] - next_maximum);",
            "block_weighted = block_weighted + scores[key_slot] * vec4<f32>(value_cache[key_slot * MAX_HEAD_VECTORS + vector_index]);",
            "running_denominator = running_denominator * previous_scale + block_denominator;",
            "let normalized = attention_output[vector_index] / running_denominator;",
            "output.data[query_base_vec + vector_index] = vec4<f16>(normalized);",
        ],
        KernelId::AddF16 => &["output.data[index] = left.data[index] + right.data[index];"],
        KernelId::GeluTanhF16 => &[
            "let value = input.data[index];",
            "let cubic = value * value * value;",
            "let argument = vec4<f16>(0.7978846h) * (value + vec4<f16>(0.044715h) * cubic);",
            "output.data[index] = vec4<f16>(0.5h) * value * (vec4<f16>(1.0h) + tanh(argument));",
        ],
        KernelId::VisionRope2dF16 => &[
            "let first_index = (token * params.heads + head) * params.head_dim + pair;",
            "let second_index = first_index + pair_count;",
            "let query_first = f32(query.data[first_index]);",
            "let key_second = f32(key.data[second_index]);",
            "query.data[first_index] = f16(query_first * cosine - query_second * sine);",
            "query.data[second_index] = f16(query_second * cosine + query_first * sine);",
            "key.data[first_index] = f16(key_first * cosine - key_second * sine);",
            "key.data[second_index] = f16(key_second * cosine + key_first * sine);",
        ],
        _ => return Err(format!("{kernel} is not a full FP16 vision kernel")),
    };
    for item in required {
        if !source.contains(item) {
            return Err(format!("{kernel} dataflow is missing {item:?}"));
        }
    }
    if matches!(
        kernel,
        KernelId::AddF16 | KernelId::GeluTanhF16
    ) && source.contains("vec4<f32>")
    {
        return Err(format!("{kernel} widens its arithmetic to vec4<f32>"));
    }
    Ok(())
}
