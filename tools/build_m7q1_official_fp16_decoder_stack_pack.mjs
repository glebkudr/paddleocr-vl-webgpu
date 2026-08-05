#!/usr/bin/env node
// Builds the logits-capable balanced decoder pack from the one converted
// IEEE-F16 checkpoint shared with the native MPS benchmark. Checkpoint
// weights are copied byte-for-byte; only the fixture-derived M-RoPE tables
// remain F32.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildM7q1BalancedWeightPack } from "./lib/m7q1_balanced_pack.mjs";
import { checkpointIdentity } from "./lib/m7q1_checkpoint_identity.mjs";
import {
  assembleM7q1OfficialFp16Pack,
  loadM7q1OfficialRopeTables,
} from "./lib/m7q1_official_pack_inputs.mjs";
import { readOfficialF16Tensor } from "../web/tests/support/m6e7_official_prefill_case.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e";
export const M7Q1_FP16_MODEL = path.resolve(
  process.env.PVLC_M7Q1_FP16_MODEL ??
    path.join(ROOT, "models", "fp16", REVISION, "model.safetensors"),
);
const PACK_OUT = path.resolve(
  process.env.PVLC_M7Q1_PACK_OUT ??
    path.join(
      ROOT,
      "web",
      "runner",
      "data",
      "m7q1-decoder-stack-logits-fp16-official.pvlc",
    ),
);
const COMPILER_BUILD = process.env.PVLC_COMPILER_BUILD ?? "0".repeat(64);

function fail(message) {
  throw new Error(`M7q1 official FP16 pack builder: ${message}`);
}

if (!fs.existsSync(M7Q1_FP16_MODEL)) {
  fail(
    `converted checkpoint is missing at ${M7Q1_FP16_MODEL}; ` +
      "run pvlc convert-checkpoint-fp16 first",
  );
}
if (fs.existsSync(PACK_OUT)) {
  fail(`refusing to replace existing pack at ${PACK_OUT}`);
}

const descriptor = {
  schema_version: 1,
  oracle: "official_l3",
  case_id: "official.decoder_stack_logits_fp16_00.0332",
  model_revision: REVISION,
  layers: 18,
  hidden_size: 1024,
  intermediate_size: 3072,
  query_heads: 16,
  key_value_heads: 2,
  head_dim: 128,
  query_width: 2048,
  key_value_width: 256,
  prefix_tokens: 0,
  cache_capacity: 337,
  rms_norm_epsilon: 1e-5,
  mrope_sections: [16, 24, 24],
};

console.log("reading the official F32 M-RoPE tables…");
const ropeTables = loadM7q1OfficialRopeTables(ROOT);
console.log("authenticating and assembling the shared-checkpoint balanced pack…");
const built = await assembleM7q1OfficialFp16Pack({
  modelPath: M7Q1_FP16_MODEL,
  descriptor,
  compilerBuild: COMPILER_BUILD,
  ropeTables,
  precisionProfile: "balanced",
  identifyCheckpoint: checkpointIdentity,
  readF16Tensor: readOfficialF16Tensor,
  buildBalancedWeightPack: buildM7q1BalancedWeightPack,
});

fs.mkdirSync(path.dirname(PACK_OUT), { recursive: true });
const temporary = `${PACK_OUT}.tmp-${process.pid}`;
try {
  fs.writeFileSync(temporary, built.packBytes, { flag: "wx" });
  fs.renameSync(temporary, PACK_OUT);
} finally {
  fs.rmSync(temporary, { force: true });
}

console.log(JSON.stringify({
  status: "passed",
  precision_profile: "balanced",
  checkpoint_path: M7Q1_FP16_MODEL,
  checkpoint_blake3: built.descriptor.checkpoint_blake3,
  checkpoint_bytes: built.descriptor.checkpoint_bytes,
  pack: PACK_OUT,
  pack_bytes: built.packBytes.byteLength,
  f16_checkpoint_shards: 11,
  f32_rope_table_shards: 2,
}, null, 2));
