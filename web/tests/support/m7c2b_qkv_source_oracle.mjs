// Checked-in caller-owned source oracle for M7c2b tests.
//
// These are literal reviewed WGSL bytes. This module deliberately does not
// import the runtime, pvlc-wgsl, source-report helpers, or runtime blake3.
// Digests were computed outside the Web/WASM trust boundary with Rust blake3.

const separateSources = Object.freeze({
  "add_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    length: u32,\n    padding0: u32,\n    padding1: u32,\n    padding2: u32,\n}\n@group(0) @binding(0) var<storage, read> left: F32Buffer;\n@group(0) @binding(1) var<storage, read> right: F32Buffer;\n@group(0) @binding(2) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(3) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let index = global_id.x;\n    if index >= params.length {\n        return;\n    }\n    output.data[index] = left.data[index] + right.data[index];\n}\n",
  "gelu_tanh_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    length: u32,\n    padding0: u32,\n    padding1: u32,\n    padding2: u32,\n}\n@group(0) @binding(0) var<storage, read> input: F32Buffer;\n@group(0) @binding(1) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(2) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let row_stride = select(params.length, params.padding0, params.padding0 != 0u);\n    let index = global_id.x + global_id.y * row_stride;\n    if index >= params.length {\n        return;\n    }\n    let value = input.data[index];\n    let cubic = value * value * value;\n    let argument = 0.7978846 * (value + 0.044715 * cubic);\n    if argument < -10.0 {\n        output.data[index] = -0.0;\n    } else if argument > 10.0 {\n        output.data[index] = value;\n    } else {\n        output.data[index] = 0.5 * value * (1.0 + tanh(argument));\n    }\n}\n",
  "layer_norm_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    rows: u32,\n    width: u32,\n    epsilon: f32,\n    padding: u32,\n}\n@group(0) @binding(0) var<storage, read> input: F32Buffer;\n@group(0) @binding(1) var<storage, read> weight: F32Buffer;\n@group(0) @binding(2) var<storage, read> bias: F32Buffer;\n@group(0) @binding(3) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(4) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let row = global_id.x;\n    if row >= params.rows {\n        return;\n    }\n    let row_start = row * params.width;\n    let first = input.data[row_start];\n    var all_equal = true;\n    var mean = 0.0;\n    for (var column = 0u; column < params.width; column = column + 1u) {\n        let value = input.data[row_start + column];\n        mean = mean + value;\n        if value != first {\n            all_equal = false;\n        }\n    }\n    if all_equal {\n        for (var column = 0u; column < params.width; column = column + 1u) {\n            output.data[row_start + column] = bias.data[column];\n        }\n        return;\n    }\n    mean = mean / f32(params.width);\n    var variance = 0.0;\n    for (var column = 0u; column < params.width; column = column + 1u) {\n        let centered = input.data[row_start + column] - mean;\n        variance = variance + centered * centered;\n    }\n    variance = variance / f32(params.width);\n    let inverse_stddev = 1.0 / sqrt(variance + params.epsilon);\n    for (var column = 0u; column < params.width; column = column + 1u) {\n        output.data[row_start + column] = (input.data[row_start + column] - mean) * inverse_stddev * weight.data[column] + bias.data[column];\n    }\n}\n",
  "vision_attention_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct U32Buffer {\n    data: array<u32>,\n}\nstruct Params {\n    tokens: u32,\n    heads: u32,\n    head_dim: u32,\n    segments: u32,\n}\n@group(0) @binding(0) var<storage, read> query: F32Buffer;\n@group(0) @binding(1) var<storage, read> key: F32Buffer;\n@group(0) @binding(2) var<storage, read> value: F32Buffer;\n@group(0) @binding(3) var<storage, read> cu_seqlens: U32Buffer;\n@group(0) @binding(4) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(5) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let linear = global_id.x;\n    let work_items = params.tokens * params.heads;\n    if linear >= work_items {\n        return;\n    }\n\n    let query_token = linear / params.heads;\n    let head = linear % params.heads;\n    var segment_start = 0u;\n    var segment_end = params.tokens;\n    for (var segment = 0u; segment < params.segments; segment = segment + 1u) {\n        let candidate_end = cu_seqlens.data[segment + 1u];\n        if query_token < candidate_end {\n            segment_start = cu_seqlens.data[segment];\n            segment_end = candidate_end;\n            break;\n        }\n    }\n\n    var weighted: array<f32, 72>;\n    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n        weighted[dimension] = 0.0;\n    }\n    var maximum = 0.0;\n    var denominator = 0.0;\n    var first_key = true;\n    let attention_scale = inverseSqrt(f32(params.head_dim));\n    let query_base = (query_token * params.heads + head) * params.head_dim;\n\n    for (var key_token = segment_start; key_token < segment_end; key_token = key_token + 1u) {\n        let key_base = (key_token * params.heads + head) * params.head_dim;\n        var score = 0.0;\n        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n            score = score + query.data[query_base + dimension] * key.data[key_base + dimension];\n        }\n        score = score * attention_scale;\n\n        if first_key {\n            maximum = score;\n            denominator = 1.0;\n            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n                weighted[dimension] = value.data[key_base + dimension];\n            }\n            first_key = false;\n        } else {\n            let next_maximum = max(maximum, score);\n            let previous_weight = exp(maximum - next_maximum);\n            let current_weight = exp(score - next_maximum);\n            denominator = denominator * previous_weight + current_weight;\n            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n                weighted[dimension] = weighted[dimension] * previous_weight + current_weight * value.data[key_base + dimension];\n            }\n            maximum = next_maximum;\n        }\n    }\n\n    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n        output.data[query_base + dimension] = weighted[dimension] / denominator;\n    }\n}\n",
  "vision_patch_projection_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    patch_count: u32,\n    input_width: u32,\n    output_width: u32,\n    padding: u32,\n}\n@group(0) @binding(0) var<storage, read> input: F32Buffer;\n@group(0) @binding(1) var<storage, read> weight: F32Buffer;\n@group(0) @binding(2) var<storage, read> bias: F32Buffer;\n@group(0) @binding(3) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(4) var<uniform> params: Params;\nconst PROJECTION_TILE_ROWS: u32 = 32u;\nconst PROJECTION_TILE_COLUMNS: u32 = 32u;\nconst PROJECTION_TILE_DEPTH: u32 = 32u;\nconst PROJECTION_ROWS_PER_LANE: u32 = 4u;\nconst PROJECTION_COLUMNS_PER_LANE: u32 = 4u;\nconst PROJECTION_WORKGROUP_SIZE: u32 = 64u;\nvar<workgroup> input_tile: array<f32, 1024>;\nvar<workgroup> weight_tile: array<f32, 1024>;\n@compute @workgroup_size(8, 8, 1)\nfn main(\n    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n    @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n    let local_index = local_id.y * 8u + local_id.x;\n    let local_row_base = local_id.y * PROJECTION_ROWS_PER_LANE;\n    let local_column_base = local_id.x * PROJECTION_COLUMNS_PER_LANE;\n    var initial_bias = vec4<f32>(0.0);\n    for (var output_offset = 0u; output_offset < PROJECTION_COLUMNS_PER_LANE; output_offset = output_offset + 1u) {\n        let output_column =\n            workgroup_id.x * PROJECTION_TILE_COLUMNS + local_column_base + output_offset;\n        if output_column < params.output_width {\n            initial_bias[output_offset] = bias.data[output_column];\n        }\n    }\n    var accumulator0 = initial_bias;\n    var accumulator1 = initial_bias;\n    var accumulator2 = initial_bias;\n    var accumulator3 = initial_bias;\n\n    for (var depth_base = 0u; depth_base < params.input_width; depth_base = depth_base + PROJECTION_TILE_DEPTH) {\n        for (var load_index = local_index; load_index < PROJECTION_TILE_ROWS * PROJECTION_TILE_DEPTH; load_index = load_index + PROJECTION_WORKGROUP_SIZE) {\n            let tile_row = load_index / PROJECTION_TILE_DEPTH;\n            let tile_depth = load_index % PROJECTION_TILE_DEPTH;\n            let input_row = workgroup_id.y * PROJECTION_TILE_ROWS + tile_row;\n            let input_depth = depth_base + tile_depth;\n            var loaded_input = 0.0;\n            if input_row < params.patch_count && input_depth < params.input_width {\n                loaded_input = input.data[input_row * params.input_width + input_depth];\n            }\n            input_tile[load_index] = loaded_input;\n\n            let output_column = workgroup_id.x * PROJECTION_TILE_COLUMNS + tile_row;\n            var loaded_weight = 0.0;\n            if output_column < params.output_width && input_depth < params.input_width {\n                loaded_weight =\n                    weight.data[output_column * params.input_width + input_depth];\n            }\n            weight_tile[tile_depth * PROJECTION_TILE_COLUMNS + tile_row] = loaded_weight;\n        }\n        workgroupBarrier();\n\n        for (var depth = 0u; depth < PROJECTION_TILE_DEPTH; depth = depth + 1u) {\n            let coefficients = vec4<f32>(\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 0u],\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 1u],\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 2u],\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 3u],\n            );\n            accumulator0 = fma(\n                vec4<f32>(input_tile[(local_row_base + 0u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator0,\n            );\n            accumulator1 = fma(\n                vec4<f32>(input_tile[(local_row_base + 1u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator1,\n            );\n            accumulator2 = fma(\n                vec4<f32>(input_tile[(local_row_base + 2u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator2,\n            );\n            accumulator3 = fma(\n                vec4<f32>(input_tile[(local_row_base + 3u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator3,\n            );\n        }\n        workgroupBarrier();\n    }\n\n    for (var output_row_offset = 0u; output_row_offset < PROJECTION_ROWS_PER_LANE; output_row_offset = output_row_offset + 1u) {\n        let output_row =\n            workgroup_id.y * PROJECTION_TILE_ROWS + local_row_base + output_row_offset;\n        var accumulated = accumulator0;\n        if output_row_offset == 1u {\n            accumulated = accumulator1;\n        } else if output_row_offset == 2u {\n            accumulated = accumulator2;\n        } else if output_row_offset == 3u {\n            accumulated = accumulator3;\n        }\n        if output_row < params.patch_count {\n            for (var output_offset = 0u; output_offset < PROJECTION_COLUMNS_PER_LANE; output_offset = output_offset + 1u) {\n                let output_column = workgroup_id.x * PROJECTION_TILE_COLUMNS\n                    + local_column_base + output_offset;\n                if output_column < params.output_width {\n                    output.data[output_row * params.output_width + output_column] =\n                        accumulated[output_offset];\n                }\n            }\n        }\n    }\n}\n",
  "vision_qkv_fused_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    tokens: u32,\n    input_width: u32,\n    output_width: u32,\n    plane_stride_elements: u32,\n}\n@group(0) @binding(0) var<storage, read> input: F32Buffer;\n@group(0) @binding(1) var<storage, read> query_weight: F32Buffer;\n@group(0) @binding(2) var<storage, read> query_bias: F32Buffer;\n@group(0) @binding(3) var<storage, read> key_weight: F32Buffer;\n@group(0) @binding(4) var<storage, read> key_bias: F32Buffer;\n@group(0) @binding(5) var<storage, read> value_weight: F32Buffer;\n@group(0) @binding(6) var<storage, read> value_bias: F32Buffer;\n@group(0) @binding(7) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(8) var<uniform> params: Params;\n@compute @workgroup_size(8, 8, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    var output_channel = 0u;\n    output_channel = global_id.x;\n    var token = 0u;\n    token = global_id.y;\n    var projection = 0u;\n    projection = global_id.z;\n    if token >= params.tokens || output_channel >= params.output_width || projection >= 3u {\n        return;\n    }\n\n    var accumulator = value_bias.data[output_channel];\n    if projection == 0u {\n        accumulator = query_bias.data[output_channel];\n    } else if projection == 1u {\n        accumulator = key_bias.data[output_channel];\n    }\n    for (var depth = 0u; depth < params.input_width; depth = depth + 1u) {\n        var coefficient = value_weight.data[output_channel * params.input_width + depth];\n        if projection == 0u {\n            coefficient = query_weight.data[output_channel * params.input_width + depth];\n        } else if projection == 1u {\n            coefficient = key_weight.data[output_channel * params.input_width + depth];\n        }\n        accumulator = accumulator + input.data[token * params.input_width + depth] * coefficient;\n    }\n    let output_index = projection * params.plane_stride_elements + token * params.output_width + output_channel;\n    output.data[output_index] = accumulator;\n}\n"
});
const separateBlake3 = Object.freeze({
  "add_f32": "c8f773daaa1634b63ed56d6a27b05927d3daff012c72258465986b39c4b7e999",
  "gelu_tanh_f32": "043067fb7d6862a1bcd742b34854b2f75259e3b894b50c848af1bcf033b67423",
  "layer_norm_f32": "315a19fc8e666a9ce51fabc75e37830fe4a1f509449fc63ad65f850255f9c917",
  "vision_attention_f32": "cc1c6e14866249b4a97bac1ceb278e7a193d75696ad615cc828d25dfe40ec3ce",
  "vision_patch_projection_f32": "c7708f8bea289e5cd54706a849102751ddc4f3b964c8d6f6ba71a92a0ce49062",
  "vision_qkv_fused_f32": "a8067e4bb517e0e1c0faa455f76f65deb5fc8b53e1af347d86ab1002d5a7d4f3"
});
const staticSources = Object.freeze({
  "add_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    length: u32,\n    padding0: u32,\n    padding1: u32,\n    padding2: u32,\n}\n@group(0) @binding(0) var<storage, read_write> left: F32Buffer;\n@group(0) @binding(1) var<storage, read_write> right: F32Buffer;\n@group(0) @binding(2) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(3) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let index = global_id.x;\n    if index >= params.length {\n        return;\n    }\n    output.data[index] = left.data[index] + right.data[index];\n}\n",
  "gelu_tanh_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    length: u32,\n    padding0: u32,\n    padding1: u32,\n    padding2: u32,\n}\n@group(0) @binding(0) var<storage, read_write> input: F32Buffer;\n@group(0) @binding(1) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(2) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let row_stride = select(params.length, params.padding0, params.padding0 != 0u);\n    let index = global_id.x + global_id.y * row_stride;\n    if index >= params.length {\n        return;\n    }\n    let value = input.data[index];\n    let cubic = value * value * value;\n    let argument = 0.7978846 * (value + 0.044715 * cubic);\n    if argument < -10.0 {\n        output.data[index] = -0.0;\n    } else if argument > 10.0 {\n        output.data[index] = value;\n    } else {\n        output.data[index] = 0.5 * value * (1.0 + tanh(argument));\n    }\n}\n",
  "layer_norm_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    rows: u32,\n    width: u32,\n    epsilon: f32,\n    padding: u32,\n}\n@group(0) @binding(0) var<storage, read_write> input: F32Buffer;\n@group(0) @binding(1) var<storage, read_write> weight: F32Buffer;\n@group(0) @binding(2) var<storage, read_write> bias: F32Buffer;\n@group(0) @binding(3) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(4) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let row = global_id.x;\n    if row >= params.rows {\n        return;\n    }\n    let row_start = row * params.width;\n    let first = input.data[row_start];\n    var all_equal = true;\n    var mean = 0.0;\n    for (var column = 0u; column < params.width; column = column + 1u) {\n        let value = input.data[row_start + column];\n        mean = mean + value;\n        if value != first {\n            all_equal = false;\n        }\n    }\n    if all_equal {\n        for (var column = 0u; column < params.width; column = column + 1u) {\n            output.data[row_start + column] = bias.data[column];\n        }\n        return;\n    }\n    mean = mean / f32(params.width);\n    var variance = 0.0;\n    for (var column = 0u; column < params.width; column = column + 1u) {\n        let centered = input.data[row_start + column] - mean;\n        variance = variance + centered * centered;\n    }\n    variance = variance / f32(params.width);\n    let inverse_stddev = 1.0 / sqrt(variance + params.epsilon);\n    for (var column = 0u; column < params.width; column = column + 1u) {\n        output.data[row_start + column] = (input.data[row_start + column] - mean) * inverse_stddev * weight.data[column] + bias.data[column];\n    }\n}\n",
  "vision_attention_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct U32Buffer {\n    data: array<u32>,\n}\nstruct Params {\n    tokens: u32,\n    heads: u32,\n    head_dim: u32,\n    segments: u32,\n}\n@group(0) @binding(0) var<storage, read_write> query: F32Buffer;\n@group(0) @binding(1) var<storage, read_write> key: F32Buffer;\n@group(0) @binding(2) var<storage, read_write> value: F32Buffer;\n@group(0) @binding(3) var<storage, read_write> cu_seqlens: U32Buffer;\n@group(0) @binding(4) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(5) var<uniform> params: Params;\n@compute @workgroup_size(64, 1, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let linear = global_id.x;\n    let work_items = params.tokens * params.heads;\n    if linear >= work_items {\n        return;\n    }\n\n    let query_token = linear / params.heads;\n    let head = linear % params.heads;\n    var segment_start = 0u;\n    var segment_end = params.tokens;\n    for (var segment = 0u; segment < params.segments; segment = segment + 1u) {\n        let candidate_end = cu_seqlens.data[segment + 1u];\n        if query_token < candidate_end {\n            segment_start = cu_seqlens.data[segment];\n            segment_end = candidate_end;\n            break;\n        }\n    }\n\n    var weighted: array<f32, 72>;\n    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n        weighted[dimension] = 0.0;\n    }\n    var maximum = 0.0;\n    var denominator = 0.0;\n    var first_key = true;\n    let attention_scale = inverseSqrt(f32(params.head_dim));\n    let query_base = (query_token * params.heads + head) * params.head_dim;\n\n    for (var key_token = segment_start; key_token < segment_end; key_token = key_token + 1u) {\n        let key_base = (key_token * params.heads + head) * params.head_dim;\n        var score = 0.0;\n        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n            score = score + query.data[query_base + dimension] * key.data[key_base + dimension];\n        }\n        score = score * attention_scale;\n\n        if first_key {\n            maximum = score;\n            denominator = 1.0;\n            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n                weighted[dimension] = value.data[key_base + dimension];\n            }\n            first_key = false;\n        } else {\n            let next_maximum = max(maximum, score);\n            let previous_weight = exp(maximum - next_maximum);\n            let current_weight = exp(score - next_maximum);\n            denominator = denominator * previous_weight + current_weight;\n            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n                weighted[dimension] = weighted[dimension] * previous_weight + current_weight * value.data[key_base + dimension];\n            }\n            maximum = next_maximum;\n        }\n    }\n\n    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {\n        output.data[query_base + dimension] = weighted[dimension] / denominator;\n    }\n}\n",
  "vision_patch_projection_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    patch_count: u32,\n    input_width: u32,\n    output_width: u32,\n    padding: u32,\n}\n@group(0) @binding(0) var<storage, read_write> input: F32Buffer;\n@group(0) @binding(1) var<storage, read_write> weight: F32Buffer;\n@group(0) @binding(2) var<storage, read_write> bias: F32Buffer;\n@group(0) @binding(3) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(4) var<uniform> params: Params;\nconst PROJECTION_TILE_ROWS: u32 = 32u;\nconst PROJECTION_TILE_COLUMNS: u32 = 32u;\nconst PROJECTION_TILE_DEPTH: u32 = 32u;\nconst PROJECTION_ROWS_PER_LANE: u32 = 4u;\nconst PROJECTION_COLUMNS_PER_LANE: u32 = 4u;\nconst PROJECTION_WORKGROUP_SIZE: u32 = 64u;\nvar<workgroup> input_tile: array<f32, 1024>;\nvar<workgroup> weight_tile: array<f32, 1024>;\n@compute @workgroup_size(8, 8, 1)\nfn main(\n    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n    @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n    let local_index = local_id.y * 8u + local_id.x;\n    let local_row_base = local_id.y * PROJECTION_ROWS_PER_LANE;\n    let local_column_base = local_id.x * PROJECTION_COLUMNS_PER_LANE;\n    var initial_bias = vec4<f32>(0.0);\n    for (var output_offset = 0u; output_offset < PROJECTION_COLUMNS_PER_LANE; output_offset = output_offset + 1u) {\n        let output_column =\n            workgroup_id.x * PROJECTION_TILE_COLUMNS + local_column_base + output_offset;\n        if output_column < params.output_width {\n            initial_bias[output_offset] = bias.data[output_column];\n        }\n    }\n    var accumulator0 = initial_bias;\n    var accumulator1 = initial_bias;\n    var accumulator2 = initial_bias;\n    var accumulator3 = initial_bias;\n\n    for (var depth_base = 0u; depth_base < params.input_width; depth_base = depth_base + PROJECTION_TILE_DEPTH) {\n        for (var load_index = local_index; load_index < PROJECTION_TILE_ROWS * PROJECTION_TILE_DEPTH; load_index = load_index + PROJECTION_WORKGROUP_SIZE) {\n            let tile_row = load_index / PROJECTION_TILE_DEPTH;\n            let tile_depth = load_index % PROJECTION_TILE_DEPTH;\n            let input_row = workgroup_id.y * PROJECTION_TILE_ROWS + tile_row;\n            let input_depth = depth_base + tile_depth;\n            var loaded_input = 0.0;\n            if input_row < params.patch_count && input_depth < params.input_width {\n                loaded_input = input.data[input_row * params.input_width + input_depth];\n            }\n            input_tile[load_index] = loaded_input;\n\n            let output_column = workgroup_id.x * PROJECTION_TILE_COLUMNS + tile_row;\n            var loaded_weight = 0.0;\n            if output_column < params.output_width && input_depth < params.input_width {\n                loaded_weight =\n                    weight.data[output_column * params.input_width + input_depth];\n            }\n            weight_tile[tile_depth * PROJECTION_TILE_COLUMNS + tile_row] = loaded_weight;\n        }\n        workgroupBarrier();\n\n        for (var depth = 0u; depth < PROJECTION_TILE_DEPTH; depth = depth + 1u) {\n            let coefficients = vec4<f32>(\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 0u],\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 1u],\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 2u],\n                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 3u],\n            );\n            accumulator0 = fma(\n                vec4<f32>(input_tile[(local_row_base + 0u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator0,\n            );\n            accumulator1 = fma(\n                vec4<f32>(input_tile[(local_row_base + 1u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator1,\n            );\n            accumulator2 = fma(\n                vec4<f32>(input_tile[(local_row_base + 2u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator2,\n            );\n            accumulator3 = fma(\n                vec4<f32>(input_tile[(local_row_base + 3u) * PROJECTION_TILE_DEPTH + depth]),\n                coefficients,\n                accumulator3,\n            );\n        }\n        workgroupBarrier();\n    }\n\n    for (var output_row_offset = 0u; output_row_offset < PROJECTION_ROWS_PER_LANE; output_row_offset = output_row_offset + 1u) {\n        let output_row =\n            workgroup_id.y * PROJECTION_TILE_ROWS + local_row_base + output_row_offset;\n        var accumulated = accumulator0;\n        if output_row_offset == 1u {\n            accumulated = accumulator1;\n        } else if output_row_offset == 2u {\n            accumulated = accumulator2;\n        } else if output_row_offset == 3u {\n            accumulated = accumulator3;\n        }\n        if output_row < params.patch_count {\n            for (var output_offset = 0u; output_offset < PROJECTION_COLUMNS_PER_LANE; output_offset = output_offset + 1u) {\n                let output_column = workgroup_id.x * PROJECTION_TILE_COLUMNS\n                    + local_column_base + output_offset;\n                if output_column < params.output_width {\n                    output.data[output_row * params.output_width + output_column] =\n                        accumulated[output_offset];\n                }\n            }\n        }\n    }\n}\n",
  "vision_qkv_fused_f32": "struct F32Buffer {\n    data: array<f32>,\n}\nstruct Params {\n    tokens: u32,\n    input_width: u32,\n    output_width: u32,\n    plane_stride_elements: u32,\n}\n@group(0) @binding(0) var<storage, read_write> input: F32Buffer;\n@group(0) @binding(1) var<storage, read_write> query_weight: F32Buffer;\n@group(0) @binding(2) var<storage, read_write> query_bias: F32Buffer;\n@group(0) @binding(3) var<storage, read_write> key_weight: F32Buffer;\n@group(0) @binding(4) var<storage, read_write> key_bias: F32Buffer;\n@group(0) @binding(5) var<storage, read_write> value_weight: F32Buffer;\n@group(0) @binding(6) var<storage, read_write> value_bias: F32Buffer;\n@group(0) @binding(7) var<storage, read_write> output: F32Buffer;\n@group(0) @binding(8) var<uniform> params: Params;\n@compute @workgroup_size(8, 8, 1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    var output_channel = 0u;\n    output_channel = global_id.x;\n    var token = 0u;\n    token = global_id.y;\n    var projection = 0u;\n    projection = global_id.z;\n    if token >= params.tokens || output_channel >= params.output_width || projection >= 3u {\n        return;\n    }\n\n    var accumulator = value_bias.data[output_channel];\n    if projection == 0u {\n        accumulator = query_bias.data[output_channel];\n    } else if projection == 1u {\n        accumulator = key_bias.data[output_channel];\n    }\n    for (var depth = 0u; depth < params.input_width; depth = depth + 1u) {\n        var coefficient = value_weight.data[output_channel * params.input_width + depth];\n        if projection == 0u {\n            coefficient = query_weight.data[output_channel * params.input_width + depth];\n        } else if projection == 1u {\n            coefficient = key_weight.data[output_channel * params.input_width + depth];\n        }\n        accumulator = accumulator + input.data[token * params.input_width + depth] * coefficient;\n    }\n    let output_index = projection * params.plane_stride_elements + token * params.output_width + output_channel;\n    output.data[output_index] = accumulator;\n}\n"
});
const staticBlake3 = Object.freeze({
  "add_f32": "783461a92ca1930b65d36c385f28bf695e70321a7f4d5fac20dbfa028b39882d",
  "gelu_tanh_f32": "6e8793a16683a469ed3b51da5015b6e8a731970a320f84b494205ab49f4a89b2",
  "layer_norm_f32": "a2746dcc4e84b7ef93fd0d0605c22369a75b19516c4b9b613c06bffb1fadcb93",
  "vision_attention_f32": "546f51f3e15008a1176c3a4d851a3a7e06e542f0b0a6d77583a79a39b1a2c393",
  "vision_patch_projection_f32": "66a9104038542a201104889146f4b28eface4ef7023c2937a6d948fb6b7642f3",
  "vision_qkv_fused_f32": "67914360405ce52eb9418787a4aa26c6469029ec4436e0f937532c7472a9871a"
});

export const M7C2B_QKV_SOURCE_ORACLE = Object.freeze({
  separate_buffers: Object.freeze({
    sources: separateSources,
    blake3: separateBlake3,
  }),
  static_arena_no_alias: Object.freeze({
    sources: staticSources,
    blake3: staticBlake3,
  }),
  static_arena_alias: Object.freeze({
    sources: staticSources,
    blake3: staticBlake3,
  }),
});

// Independent test-only BLAKE3. It intentionally shares no code with the
// runtime/WASM implementation whose output this oracle verifies.
const BLAKE3_IV = Object.freeze([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);
const BLAKE3_MESSAGE_PERMUTATION = Object.freeze([
  2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8,
]);
const CHUNK_START = 1;
const CHUNK_END = 2;
const PARENT = 4;
const ROOT = 8;
const ORACLE_REFLECT_APPLY = Reflect.apply;
const ORACLE_MATH_CEIL = Math.ceil;
const ORACLE_MATH_FLOOR = Math.floor;
const ORACLE_MATH_MAX = Math.max;
const ORACLE_MATH_MIN = Math.min;
const ORACLE_NUMBER_TO_STRING = Number.prototype.toString;

function rotateRight(value, count) {
  return ((value >>> count) | (value << (32 - count))) >>> 0;
}

function mix(state, a, b, c, d, left, right) {
  state[a] = (state[a] + state[b] + left) >>> 0;
  state[d] = rotateRight(state[d] ^ state[a], 16);
  state[c] = (state[c] + state[d]) >>> 0;
  state[b] = rotateRight(state[b] ^ state[c], 12);
  state[a] = (state[a] + state[b] + right) >>> 0;
  state[d] = rotateRight(state[d] ^ state[a], 8);
  state[c] = (state[c] + state[d]) >>> 0;
  state[b] = rotateRight(state[b] ^ state[c], 7);
}

function round(state, message) {
  mix(state, 0, 4, 8, 12, message[0], message[1]);
  mix(state, 1, 5, 9, 13, message[2], message[3]);
  mix(state, 2, 6, 10, 14, message[4], message[5]);
  mix(state, 3, 7, 11, 15, message[6], message[7]);
  mix(state, 0, 5, 10, 15, message[8], message[9]);
  mix(state, 1, 6, 11, 12, message[10], message[11]);
  mix(state, 2, 7, 8, 13, message[12], message[13]);
  mix(state, 3, 4, 9, 14, message[14], message[15]);
}

function blockWords(bytes) {
  const words = new Uint32Array(16);
  for (let index = 0; index < bytes.length; index += 1) {
    words[index >>> 2] |= bytes[index] << (8 * (index & 3));
  }
  return words;
}

function compress(inputCv, inputWords, counter, blockLength, flags) {
  const state = new Uint32Array(16);
  state.set(inputCv, 0);
  state.set(BLAKE3_IV.slice(0, 4), 8);
  state[12] = counter >>> 0;
  state[13] = ORACLE_MATH_FLOOR(counter / 0x1_0000_0000) >>> 0;
  state[14] = blockLength;
  state[15] = flags;
  let message = Array.from(inputWords);
  for (let roundIndex = 0; roundIndex < 7; roundIndex += 1) {
    round(state, message);
    message = BLAKE3_MESSAGE_PERMUTATION.map((index) => message[index]);
  }
  const output = new Uint32Array(16);
  for (let index = 0; index < 8; index += 1) {
    output[index] = state[index] ^ state[index + 8];
    output[index + 8] = state[index + 8] ^ inputCv[index];
  }
  return output;
}

function output(inputCv, words, counter, blockLength, flags) {
  return {
    chainingValue() {
      return compress(inputCv, words, counter, blockLength, flags).slice(0, 8);
    },
    rootBytes() {
      const root = compress(inputCv, words, 0, blockLength, flags | ROOT);
      const bytes = new Uint8Array(32);
      for (let index = 0; index < 8; index += 1) {
        const word = root[index];
        bytes[index * 4] = word & 0xff;
        bytes[index * 4 + 1] = (word >>> 8) & 0xff;
        bytes[index * 4 + 2] = (word >>> 16) & 0xff;
        bytes[index * 4 + 3] = word >>> 24;
      }
      return bytes;
    },
  };
}

function chunkOutput(chunk, chunkCounter) {
  const blockCount = ORACLE_MATH_MAX(1, ORACLE_MATH_CEIL(chunk.length / 64));
  let chainingValue = Uint32Array.from(BLAKE3_IV);
  for (let blockIndex = 0; blockIndex < blockCount - 1; blockIndex += 1) {
    const words = blockWords(chunk.subarray(blockIndex * 64, blockIndex * 64 + 64));
    const flags = blockIndex === 0 ? CHUNK_START : 0;
    chainingValue = compress(chainingValue, words, chunkCounter, 64, flags).slice(0, 8);
  }
  const finalStart = (blockCount - 1) * 64;
  const finalBlock = chunk.subarray(finalStart, ORACLE_MATH_MIN(finalStart + 64, chunk.length));
  const flags = CHUNK_END | (blockCount === 1 ? CHUNK_START : 0);
  return output(chainingValue, blockWords(finalBlock), chunkCounter, finalBlock.length, flags);
}

function parentOutput(left, right) {
  const words = new Uint32Array(16);
  words.set(left.chainingValue(), 0);
  words.set(right.chainingValue(), 8);
  return output(Uint32Array.from(BLAKE3_IV), words, 0, 64, PARENT);
}

export function callerOwnedBlake3Hex(value) {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  if (!(bytes instanceof Uint8Array)) throw new TypeError("BLAKE3 input must be a string or Uint8Array");
  let level = [];
  for (let offset = 0, chunkCounter = 0; offset < ORACLE_MATH_MAX(1, bytes.length); offset += 1024) {
    level.push(chunkOutput(
      bytes.subarray(offset, ORACLE_MATH_MIN(offset + 1024, bytes.length)),
      chunkCounter,
    ));
    chunkCounter += 1;
  }
  while (level.length > 1) {
    const next = [];
    for (let index = 0; index < level.length; index += 2) {
      next.push(index + 1 < level.length ? parentOutput(level[index], level[index + 1]) : level[index]);
    }
    level = next;
  }
  return Array.from(level[0].rootBytes(), (byte) =>
    ORACLE_REFLECT_APPLY(ORACLE_NUMBER_TO_STRING, byte, [16]).padStart(2, "0")).join("");
}
