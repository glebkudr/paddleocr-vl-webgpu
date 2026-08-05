#!/usr/bin/env node
// Builds full-FP16 browser vision-encoder shards from the exact IEEE-F16
// checkpoint shared with the native MPS benchmark.
//
// The six large matrices in every encoder layer preserve their exact F16 bits
// while being transposed once into the input-major layout consumed
// coalescently by adjacent WebGPU lanes. LayerNorm vectors and linear biases
// preserve their original F16 bytes.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { checkpointIdentity } from "./lib/m7q1_checkpoint_identity.mjs";
import {
  VISION_LAYER_TENSOR_ROLES,
  materializeVisionTensor,
} from "./lib/fp16_matrix_layout.mjs";
import { readOfficialF16Tensor } from "../web/tests/support/m6e7_official_prefill_case.mjs";
import { callerOwnedBlake3Hex } from "../web/tests/support/m7c2b_qkv_source_oracle.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const EXPECTED_CHECKPOINT_BLAKE3 =
  "7eaf17cbadb7ff816491a3bfe8c96abc52c85ceea5506e73f3eb676cff57655a";
const EXPECTED_CHECKPOINT_BYTES = 1_917_255_480;
const MODEL = path.resolve(
  process.env.PVLC_FP16_MODEL ??
    path.join(ROOT, "models", "fp16", REVISION, "model.safetensors"),
);
const OUTPUT = path.resolve(
  process.env.PVLC_FP16_VISION_OUT ??
    path.join(
      ROOT,
      "web",
      "runner",
      "data",
      "m7u-vision-stack-full-fp16-input-major-official",
    ),
);

const HIDDEN = 1152;
const INTERMEDIATE = 4304;
const LAYERS = 27;
const TOKENS = 1740;
const LAYER_WEIGHT_BYTES = 30_479_008;
const POST_NORM_BYTES = 4_608;
const INPUT_BYTES = TOKENS * HIDDEN * 2;
const TRANSPORT_BYTES =
  INPUT_BYTES + LAYERS * LAYER_WEIGHT_BYTES + POST_NORM_BYTES;

function invariant(condition, message) {
  if (!condition) {
    throw new Error(`FP16 vision-stack builder: ${message}`);
  }
}

function concatenate(parts, label) {
  const total = parts.reduce((bytes, part) => bytes + part.byteLength, 0);
  invariant(Number.isSafeInteger(total) && total > 0, `${label} byte length is invalid`);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    invariant(part instanceof Uint8Array, `${label} contains a non-byte tensor`);
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

function materializeVector(name, shape) {
  const raw = readOfficialF16Tensor(MODEL, name, shape);
  return materializeVisionTensor({
    raw,
    shape,
    storage: "f16",
    label: name,
  });
}

function descriptor(id, kind, layerIndex, payload) {
  return {
    id,
    kind,
    layer_index: layerIndex,
    bytes: payload.byteLength,
    blake3: callerOwnedBlake3Hex(payload),
  };
}

function writeExclusive(file, bytes) {
  fs.writeFileSync(file, bytes, { flag: "wx" });
}

invariant(fs.existsSync(MODEL), `shared checkpoint is missing at ${MODEL}`);
invariant(!fs.existsSync(OUTPUT), `refusing to replace existing output ${OUTPUT}`);

const identity = await checkpointIdentity(MODEL);
invariant(
  identity.checkpoint_blake3 === EXPECTED_CHECKPOINT_BLAKE3 &&
    identity.checkpoint_bytes === EXPECTED_CHECKPOINT_BYTES &&
    identity.dtype === "float16",
  "shared checkpoint identity drifted from the native/decoder FP16 source",
);

fs.mkdirSync(OUTPUT, { recursive: false });
const shards = [
  {
    id: "input.embeddings",
    kind: "input",
    layer_index: null,
    bytes: INPUT_BYTES,
    // This template input is replaced by the authenticated per-image
    // embedding descriptor before the runtime sees the manifest.
    blake3: "0".repeat(64),
  },
];

try {
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const prefix = `visual.vision_model.encoder.layers.${layer}`;
    const parts = VISION_LAYER_TENSOR_ROLES.map(([suffix, shape]) =>
      materializeVisionTensor({
        raw: readOfficialF16Tensor(MODEL, `${prefix}.${suffix}`, shape),
        shape,
        storage: "f16",
        label: `${prefix}.${suffix}`,
      })
    );
    const payload = concatenate(parts, `vision layer ${layer}`);
    invariant(
      payload.byteLength === LAYER_WEIGHT_BYTES,
      `vision layer ${layer} has ${payload.byteLength} bytes, expected ${LAYER_WEIGHT_BYTES}`,
    );
    const id = `weights.vision_layer.${String(layer).padStart(2, "0")}`;
    writeExclusive(path.join(OUTPUT, `${id}.bin`), payload);
    shards.push(descriptor(id, "layer", layer, payload));
    console.log(`wrote ${id} (${payload.byteLength} bytes)`);
  }

  const postNorm = concatenate([
    materializeVector(
      "visual.vision_model.post_layernorm.weight",
      [HIDDEN],
    ),
    materializeVector(
      "visual.vision_model.post_layernorm.bias",
      [HIDDEN],
    ),
  ], "vision post-norm");
  invariant(
    postNorm.byteLength === POST_NORM_BYTES,
    "vision post-norm byte length drifted",
  );
  const postNormId = "weights.vision_post_norm";
  writeExclusive(path.join(OUTPUT, `${postNormId}.bin`), postNorm);
  shards.push(descriptor(postNormId, "post_norm", null, postNorm));

  const manifest = {
    schema_version: 1,
    oracle: "synthetic",
    case_id: "synthetic.shared_checkpoint_fp16/vision.stack.27",
    model_id: "PaddlePaddle/PaddleOCR-VL-1.6",
    model_revision: REVISION,
    compiler_model_abi: 1,
    compiler_build: "0".repeat(64),
    golden_bundle_digest: null,
    semantic_fingerprint: null,
    matrix_weight_storage: "f16",
    matrix_weight_layout: "input_major",
    vector_weight_storage: "f16",
    activation_storage: "f16",
    tokens: TOKENS,
    hidden_size: HIDDEN,
    attention_heads: 16,
    head_dim: 72,
    intermediate_size: INTERMEDIATE,
    layer_norm_epsilon: 0.000001,
    cu_seqlens: [0, TOKENS],
    layer_count: LAYERS,
    checkpoint_layers: [],
    shards,
  };
  writeExclusive(
    path.join(OUTPUT, "manifest.json"),
    `${JSON.stringify(manifest)}\n`,
  );
  writeExclusive(
    path.join(OUTPUT, "checkpoint.json"),
    `${JSON.stringify({
      schema_version: 1,
      checkpoint_blake3: identity.checkpoint_blake3,
      checkpoint_bytes: identity.checkpoint_bytes,
      checkpoint_dtype: identity.dtype,
      matrix_weight_storage: "f16",
      matrix_weight_layout: "input_major",
      vector_weight_storage: "f16",
      activation_storage: "f16",
      layer_weight_bytes: LAYER_WEIGHT_BYTES,
      transport_bytes: TRANSPORT_BYTES,
    })}\n`,
  );

  console.log(JSON.stringify({
    status: "passed",
    output: OUTPUT,
    checkpoint_blake3: identity.checkpoint_blake3,
    checkpoint_bytes: identity.checkpoint_bytes,
    layer_weight_bytes: LAYER_WEIGHT_BYTES,
    transport_bytes: TRANSPORT_BYTES,
  }, null, 2));
} catch (error) {
  // Preserve any completed shard files for forensic inspection. The output
  // directory is never treated as valid without the manifest commit record.
  throw error;
}
