#!/usr/bin/env node
// Builds the M6e7 official browser evidence artifacts:
//
//   web/runner/data/m6e7-decoder-stack-prefill-official.pvlc
//     The exact PVLCPK01 decoder stack session weight pack for the official
//     332-token prefill case: the eleven shard payloads are the pinned BF16
//     model weights of models/snapshots/66317acc…/model.safetensors
//     converted to f32 (BF16 -> f32 is an exact bit shift), laid out as the
//     accepted 18x layer-major bulks, plus the axis-major M-RoPE tables
//     [3, 337, 128] whose rows [0, 332) are the official
//     decoder.rope.{cos,sin}.axis_major fixture values and whose rows
//     [332, 337) are zero (the official case runs prefill + finish only, so
//     those rows are never read; decode continuation at official scale is
//     proven on the synthetic ladder).
//   web/runner/data/m6e7-decoder-stack-prefill-official.safetensors
//     The f32 expectation bundle consumed by the browser page:
//     hidden_states [332, 1024]      = decoder.layer.00.input (BF16 -> f32)
//     expected_final_row [1024]      = decoder.layer.17.output row 331
//     expected_key_cache [18, 332, 256] / expected_value_cache [18, 332, 256]
//       = the [0, 332) prefix slices of the native decode KV caches
//         (decoder.decode.00.kv.{key,value}.layer_token_major, BF16 -> f32)
//
// The pack is assembled by the accepted M6e6 PVLCPK01 builder authority
// (buildM6e6WeightPack), so the browser begin() admission path is identical
// to the synthetic gate. Both artifacts are regenerable outputs (the
// web/runner/data directory is git-ignored) and are verified after writing:
// the pack is re-parsed by validateM6e6WeightPack with the prefill admission
// and the bundle is re-parsed for exact geometry.
//
// Usage: node tools/build_m6e7_official_decoder_stack_pack.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildM6e6WeightPack,
  validateM6e6WeightPack,
} from "../web/tests/support/m6e6_decoder_stack_session_oracle.mjs";
import { callerOwnedBlake3Hex } from "../web/tests/support/m7c2b_qkv_source_oracle.mjs";
import {
  M6E7_OFFICIAL_CAPACITY,
  M6E7_OFFICIAL_HIDDEN,
  M6E7_OFFICIAL_KV_WIDTH,
  M6E7_OFFICIAL_LAYERS,
  M6E7_OFFICIAL_PATHS,
  M6E7_OFFICIAL_TOKENS,
  loadM6e7OfficialPrefillCase,
} from "../web/tests/support/m6e7_official_prefill_case.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PACK_OUT = path.join(ROOT, M6E7_OFFICIAL_PATHS.packOut);
const EXPECTATIONS_OUT = path.join(ROOT, M6E7_OFFICIAL_PATHS.expectationsOut);

const LAYERS = M6E7_OFFICIAL_LAYERS;
const HIDDEN = M6E7_OFFICIAL_HIDDEN;
const KV_WIDTH = M6E7_OFFICIAL_KV_WIDTH;
const TOKENS = M6E7_OFFICIAL_TOKENS;
const CAPACITY = M6E7_OFFICIAL_CAPACITY;

function fail(message) {
  console.error(`m6e7 official pack builder: ${message}`);
  process.exit(1);
}

console.log("reading the official prefill case (model, stack fixture, decode fixture)…");
const official = loadM6e7OfficialPrefillCase(ROOT);
const {
  hiddenStates,
  expectedFinalRow,
  expectedKeyCache,
  expectedValueCache,
} = official;
console.log("assembling the PVLCPK01 session weight pack…");
const packCase = {
  descriptor: official.descriptor,
  weights: {
    ...official.weights,
    ropeCos: official.ropeCos,
    ropeSin: official.ropeSin,
  },
};
const { packBytes } = buildM6e6WeightPack(packCase, {
  oracle: "official_l3",
  caseId: "official.decoder_stack_prefill_00.0332",
});

console.log("assembling the f32 expectation bundle…");
function writeSafetensors(tensors) {
  const header = {};
  let offset = 0;
  const parts = [];
  for (const [name, { shape, values }] of Object.entries(tensors)) {
    const bytes = values.byteLength;
    header[name] = {
      dtype: "F32",
      shape,
      data_offsets: [offset, offset + bytes],
    };
    parts.push(new Uint8Array(values.buffer, values.byteOffset, bytes));
    offset += bytes;
  }
  // Pad the JSON header with trailing spaces (legal JSON whitespace) so the
  // tensor data base stays 8-byte aligned for zero-copy typed-array views.
  let headerText = JSON.stringify(header);
  const misalignment = (8 + Buffer.byteLength(headerText, "utf8")) % 8;
  if (misalignment !== 0) {
    headerText += " ".repeat(8 - misalignment);
  }
  const headerBytes = Buffer.from(headerText, "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.byteLength), 0);
  return Buffer.concat([prefix, headerBytes, ...parts.map((part) => Buffer.from(part))]);
}
const expectationsBytes = writeSafetensors({
  hidden_states: { shape: [TOKENS, HIDDEN], values: hiddenStates },
  expected_final_row: { shape: [HIDDEN], values: expectedFinalRow },
  expected_key_cache: {
    shape: [LAYERS, TOKENS, KV_WIDTH],
    values: expectedKeyCache,
  },
  expected_value_cache: {
    shape: [LAYERS, TOKENS, KV_WIDTH],
    values: expectedValueCache,
  },
});

fs.mkdirSync(path.dirname(PACK_OUT), { recursive: true });
fs.writeFileSync(PACK_OUT, packBytes);
fs.writeFileSync(EXPECTATIONS_OUT, expectationsBytes);

console.log("verifying the written artifacts…");
const writtenPack = new Uint8Array(fs.readFileSync(PACK_OUT));
const validated = validateM6e6WeightPack(writtenPack, {
  cacheCapacity: CAPACITY,
  prefixTokens: 0,
  prefillCapable: true,
  oracle: "official_l3",
  caseId: "official.decoder_stack_prefill_00.0332",
});
if (validated.descriptor.prefix_tokens !== 0 ||
    validated.descriptor.cache_capacity !== CAPACITY ||
    validated.descriptor.oracle !== "official_l3") {
  fail("written pack descriptor drifted");
}
const bundle = fs.readFileSync(EXPECTATIONS_OUT);
const bundleHeaderLength = Number(bundle.readBigUInt64LE(0));
const bundleHeader = JSON.parse(
  bundle.subarray(8, 8 + bundleHeaderLength).toString("utf8"),
);
for (const [name, shape] of [
  ["hidden_states", [TOKENS, HIDDEN]],
  ["expected_final_row", [HIDDEN]],
  ["expected_key_cache", [LAYERS, TOKENS, KV_WIDTH]],
  ["expected_value_cache", [LAYERS, TOKENS, KV_WIDTH]],
]) {
  const info = bundleHeader[name];
  if (info === undefined || info.dtype !== "F32" ||
      JSON.stringify(info.shape) !== JSON.stringify(shape)) {
    fail(`written bundle tensor ${name} drifted`);
  }
}

console.log(JSON.stringify({
  status: "passed",
  pack: path.relative(ROOT, PACK_OUT),
  pack_bytes: writtenPack.byteLength,
  pack_blake3: callerOwnedBlake3Hex(writtenPack),
  expectations: path.relative(ROOT, EXPECTATIONS_OUT),
  expectations_bytes: bundle.byteLength,
  expectations_blake3: callerOwnedBlake3Hex(new Uint8Array(bundle)),
  tokens: TOKENS,
  cache_capacity: CAPACITY,
}, null, 2));
