#!/usr/bin/env node
// Builds the browser projector weights directly from the same IEEE-F16
// checkpoint used by the native MPS benchmark and the FP16 vision stack.
//
// Matrix payloads are transposed once from checkpoint output-major layout to
// the input-major layout consumed by LinearProjectionF16. Vector payloads keep
// their exact checkpoint bits.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { checkpointIdentity } from "./lib/m7q1_checkpoint_identity.mjs";
import { materializeVisionTensor } from "./lib/fp16_matrix_layout.mjs";
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
  process.env.PVLC_FP16_PROJECTOR_OUT ??
    path.join(
      ROOT,
      "web",
      "runner",
      "data",
      "m7v-projector-full-fp16-input-major-official",
    ),
);

const HIDDEN = 1152;
const MERGED = HIDDEN * 4;
const OUTPUT_WIDTH = 1024;
const WEIGHTS_BYTES = 51_920_384;

const TENSORS = Object.freeze([
  Object.freeze(["pre_norm_weight", "mlp_AR.pre_norm.weight", [HIDDEN]]),
  Object.freeze(["pre_norm_bias", "mlp_AR.pre_norm.bias", [HIDDEN]]),
  Object.freeze(["linear1_weight", "mlp_AR.linear_1.weight", [MERGED, MERGED]]),
  Object.freeze(["linear1_bias", "mlp_AR.linear_1.bias", [MERGED]]),
  Object.freeze(["linear2_weight", "mlp_AR.linear_2.weight", [OUTPUT_WIDTH, MERGED]]),
  Object.freeze(["linear2_bias", "mlp_AR.linear_2.bias", [OUTPUT_WIDTH]]),
]);

function invariant(condition, message) {
  if (!condition) {
    throw new Error(`FP16 projector builder: ${message}`);
  }
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
  "shared checkpoint identity drifted from the native FP16 source",
);

const parts = [];
const ranges = {};
let offset = 0;
for (const [role, tensorName, shape] of TENSORS) {
  const payload = materializeVisionTensor({
    raw: readOfficialF16Tensor(MODEL, tensorName, shape),
    shape,
    storage: "f16",
    label: tensorName,
  });
  ranges[role] = Object.freeze({ offset, bytes: payload.byteLength });
  parts.push(payload);
  offset += payload.byteLength;
}
invariant(offset === WEIGHTS_BYTES, `payload has ${offset} bytes, expected ${WEIGHTS_BYTES}`);

const weights = new Uint8Array(WEIGHTS_BYTES);
offset = 0;
for (const part of parts) {
  weights.set(part, offset);
  offset += part.byteLength;
}
const weightsBlake3 = callerOwnedBlake3Hex(weights);

fs.mkdirSync(OUTPUT, { recursive: false });
writeExclusive(path.join(OUTPUT, "weights.projector.bin"), weights);
writeExclusive(
  path.join(OUTPUT, "manifest.json"),
  `${JSON.stringify({
    schema_version: 1,
    model_id: "PaddlePaddle/PaddleOCR-VL-1.6",
    model_revision: REVISION,
    checkpoint_blake3: identity.checkpoint_blake3,
    checkpoint_bytes: identity.checkpoint_bytes,
    checkpoint_dtype: identity.dtype,
    weights_blake3: weightsBlake3,
    weights_bytes: weights.byteLength,
    weight_storage: "f16",
    matrix_weight_layout: "input_major",
    activation_storage: "f16",
    hidden_size: HIDDEN,
    output_size: OUTPUT_WIDTH,
    layer_norm_epsilon: 0.00001,
    ranges,
  })}\n`,
);

console.log(JSON.stringify({
  status: "passed",
  output: OUTPUT,
  checkpoint_blake3: identity.checkpoint_blake3,
  weights_blake3: weightsBlake3,
  weights_bytes: weights.byteLength,
}, null, 2));
