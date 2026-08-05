#!/usr/bin/env node
// Builds the M6e8 official browser evidence artifacts:
//
//   web/runner/data/m6e8-decoder-stack-logits-official.pvlc
//     The exact PVLCPK01 logits-capable decoder stack session weight pack for
//     the official 332-token case: the accepted eleven M6e7 shard payloads
//     (the pinned BF16 model weights converted to f32, the 18x layer-major
//     bulks, and the axis-major M-RoPE tables [3, 337, 128]) PLUS the two
//     M6e8 shards at the end of the pinned order —
//       weights.final_layernorm = model.norm.weight [1024] (BF16 -> f32)
//       weights.lm_head         = lm_head.weight [103424, 1024] (BF16 -> f32)
//     The raw BF16 lm_head payload is pinned by BLAKE3
//     784ffd4944c3b72292fa62a8f6044485aef55be16479ac7946eaf0e7ba3e08dc and the
//     pin is verified at build time. Unlike the M6e7 pack (prefill + finish
//     only), the M6e8 official flow continues with ONE decode step at
//     position 332, so M-RoPE table row 332 carries the official
//     decoder.decode.00.rope.{cos,sin}.axis_major values (rows [333, 337)
//     stay zero — never read by prefill + one step).
//   web/runner/data/m6e8-decoder-stack-logits-official.safetensors
//     The f32 expectation bundle consumed by the browser page:
//       hidden_states [332, 1024]      = decoder.layer.00.input (BF16 -> f32)
//       decode_step_hidden [1024]      = decoder.decode.00.layer.00.input
//       expected_prefill_logits [103424]
//                                      = decoder.prefill.logits.last
//                                        (prefill-lm-head-official-v1)
//       expected_decode_logits [103424]
//                                      = decoder.decode.00.logits
//
// The pack is assembled by the M6e8 oracle pack authority
// (buildM6e8WeightPack), so the browser begin() admission path is identical
// to the synthetic gate. Both artifacts are regenerable outputs (the
// web/runner/data directory is git-ignored) and are verified after writing:
// the pack is re-parsed by validateM6e8WeightPack with the prefill admission
// and the bundle is re-parsed for exact geometry.
//
// Usage: node tools/build_m6e8_official_decoder_stack_pack.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildM6e8WeightPack,
  validateM6e8WeightPack,
} from "../web/tests/support/m6e8_decoder_stack_logits_oracle.mjs";
import { m6e6FastBlake3Hex } from "../web/tests/support/m6e6_decoder_stack_session_oracle.mjs";
import { callerOwnedBlake3Hex } from "../web/tests/support/m7c2b_qkv_source_oracle.mjs";
import {
  M6E7_OFFICIAL_CAPACITY,
  M6E7_OFFICIAL_HEAD_DIM,
  M6E7_OFFICIAL_HIDDEN,
  M6E7_OFFICIAL_LAYERS,
  M6E7_OFFICIAL_PATHS,
  M6E7_OFFICIAL_TOKENS,
  loadM6e7OfficialPrefillCase,
  readOfficialBf16Tensor,
} from "../web/tests/support/m6e7_official_prefill_case.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PACK_OUT = path.join(ROOT, "web/runner/data/m6e8-decoder-stack-logits-official.pvlc");
const EXPECTATIONS_OUT = path.join(
  ROOT,
  "web/runner/data/m6e8-decoder-stack-logits-official.safetensors",
);
const PREFILL_LM_HEAD_FIXTURE =
  "crates/pvlc-testkit/tests/fixtures/prefill-lm-head-official-v1.safetensors";

const LAYERS = M6E7_OFFICIAL_LAYERS;
const HIDDEN = M6E7_OFFICIAL_HIDDEN;
const HEAD_DIM = M6E7_OFFICIAL_HEAD_DIM;
const TOKENS = M6E7_OFFICIAL_TOKENS;
const CAPACITY = M6E7_OFFICIAL_CAPACITY;
const VOCAB = 103424;
const LM_HEAD_RAW_BF16_BLAKE3 =
  "784ffd4944c3b72292fa62a8f6044485aef55be16479ac7946eaf0e7ba3e08dc";

function fail(message) {
  console.error(`m6e8 official pack builder: ${message}`);
  process.exit(1);
}

console.log("reading the official prefill case (model, stack fixture, decode fixture)…");
const official = loadM6e7OfficialPrefillCase(ROOT);

console.log("reading the logits operands (final norm, LM head)…");
const finalNorm = readOfficialBf16Tensor(
  ROOT,
  M6E7_OFFICIAL_PATHS.model,
  "model.norm.weight",
  [HIDDEN],
);
const lmHead = readOfficialBf16Tensor(
  ROOT,
  M6E7_OFFICIAL_PATHS.model,
  "lm_head.weight",
  [VOCAB, HIDDEN],
);
const lmHeadRawBlake3 = m6e6FastBlake3Hex(lmHead.raw);
if (lmHeadRawBlake3 !== LM_HEAD_RAW_BF16_BLAKE3) {
  fail(
    `raw BF16 lm_head.weight BLAKE3 drifted: ${lmHeadRawBlake3} ` +
      `(expected ${LM_HEAD_RAW_BF16_BLAKE3})`,
  );
}
console.log(`raw BF16 lm_head BLAKE3 pinned: ${lmHeadRawBlake3}`);

console.log("reading the decode-step evidence (hidden input, rope row, logits)…");
const decodeStepHidden = readOfficialBf16Tensor(
  ROOT,
  M6E7_OFFICIAL_PATHS.decodeFixture,
  "decoder.decode.00.layer.00.input",
  [1, HIDDEN],
);
const decodeRopeCos = readOfficialBf16Tensor(
  ROOT,
  M6E7_OFFICIAL_PATHS.decodeFixture,
  "decoder.decode.00.rope.cos.axis_major",
  [3, 1, HEAD_DIM],
);
const decodeRopeSin = readOfficialBf16Tensor(
  ROOT,
  M6E7_OFFICIAL_PATHS.decodeFixture,
  "decoder.decode.00.rope.sin.axis_major",
  [3, 1, HEAD_DIM],
);
const expectedDecodeLogits = readOfficialBf16Tensor(
  ROOT,
  M6E7_OFFICIAL_PATHS.decodeFixture,
  "decoder.decode.00.logits",
  [1, VOCAB],
);
const expectedPrefillLogits = readOfficialBf16Tensor(
  ROOT,
  PREFILL_LM_HEAD_FIXTURE,
  "decoder.prefill.logits.last",
  [1, VOCAB],
);

// The M6e8 official flow continues with one decode step at position 332, so
// M-RoPE row 332 of every axis is the official decode rope row (the M6e7 pack
// keeps it zero because its official flow never decodes).
const ropeCos = official.ropeCos.slice();
const ropeSin = official.ropeSin.slice();
for (let axis = 0; axis < 3; axis += 1) {
  ropeCos.set(
    decodeRopeCos.values.subarray(axis * HEAD_DIM, (axis + 1) * HEAD_DIM),
    (axis * CAPACITY + TOKENS) * HEAD_DIM,
  );
  ropeSin.set(
    decodeRopeSin.values.subarray(axis * HEAD_DIM, (axis + 1) * HEAD_DIM),
    (axis * CAPACITY + TOKENS) * HEAD_DIM,
  );
}

console.log("assembling the PVLCPK01 logits-capable session weight pack…");
const packCase = {
  descriptor: official.descriptor,
  weights: {
    ...official.weights,
    ropeCos,
    ropeSin,
    finalNorm: finalNorm.values,
    lmHead: lmHead.values,
  },
};
const { packBytes } = buildM6e8WeightPack(packCase, {
  oracle: "official_l3",
  caseId: "official.decoder_stack_logits_00.0332",
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
  hidden_states: { shape: [TOKENS, HIDDEN], values: official.hiddenStates },
  decode_step_hidden: { shape: [HIDDEN], values: decodeStepHidden.values },
  expected_prefill_logits: { shape: [VOCAB], values: expectedPrefillLogits.values },
  expected_decode_logits: { shape: [VOCAB], values: expectedDecodeLogits.values },
});

fs.mkdirSync(path.dirname(PACK_OUT), { recursive: true });
fs.writeFileSync(PACK_OUT, packBytes);
fs.writeFileSync(EXPECTATIONS_OUT, expectationsBytes);

console.log("verifying the written artifacts…");
const writtenPack = new Uint8Array(fs.readFileSync(PACK_OUT));
const validated = validateM6e8WeightPack(writtenPack, {
  cacheCapacity: CAPACITY,
  prefixTokens: 0,
  prefillCapable: true,
  oracle: "official_l3",
  caseId: "official.decoder_stack_logits_00.0332",
});
if (validated.descriptor.prefix_tokens !== 0 ||
    validated.descriptor.cache_capacity !== CAPACITY ||
    validated.descriptor.oracle !== "official_l3" ||
    Object.keys(validated.payloads).length !== 13) {
  fail("written pack descriptor drifted");
}
// The rope patch must round-trip: row 332 of every axis is the official
// decode rope row, row 331 stays the official prefill row, row 333 stays zero.
const writtenCos = validated.payloads["weights.mrope_cos"];
for (let axis = 0; axis < 3; axis += 1) {
  const row = writtenCos.subarray(
    (axis * CAPACITY + TOKENS) * HEAD_DIM,
    (axis * CAPACITY + TOKENS + 1) * HEAD_DIM,
  );
  const expected = decodeRopeCos.values.subarray(axis * HEAD_DIM, (axis + 1) * HEAD_DIM);
  for (let index = 0; index < HEAD_DIM; index += 1) {
    if (row[index] !== expected[index]) fail(`written pack rope cos row 332 axis ${axis} drifted`);
  }
  const tail = writtenCos.subarray(
    (axis * CAPACITY + TOKENS + 1) * HEAD_DIM,
    (axis * CAPACITY + TOKENS + 2) * HEAD_DIM,
  );
  if (!tail.every((value) => value === 0)) fail("written pack rope cos row 333 is not zero");
}
const bundle = fs.readFileSync(EXPECTATIONS_OUT);
const bundleHeaderLength = Number(bundle.readBigUInt64LE(0));
const bundleHeader = JSON.parse(
  bundle.subarray(8, 8 + bundleHeaderLength).toString("utf8"),
);
for (const [name, shape] of [
  ["hidden_states", [TOKENS, HIDDEN]],
  ["decode_step_hidden", [HIDDEN]],
  ["expected_prefill_logits", [VOCAB]],
  ["expected_decode_logits", [VOCAB]],
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
  lm_head_raw_bf16_blake3: lmHeadRawBlake3,
  tokens: TOKENS,
  cache_capacity: CAPACITY,
  vocab_size: VOCAB,
}, null, 2));
