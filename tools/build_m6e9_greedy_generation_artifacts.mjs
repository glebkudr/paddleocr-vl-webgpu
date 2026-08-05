#!/usr/bin/env node
// Builds the M6e9 browser greedy generation artifacts:
//
//   web/runner/data/m6e9-embed-tokens.f32
//     The exact BF16 -> f32 widening of the pinned checkpoint
//     model.embed_tokens.weight [103424, 1024] (BF16 -> f32 is an exact bit
//     shift; the raw BF16 BLAKE3
//     a71f8a645b457a0b1dfaf138cd22252a802731a87c7c629cb9831ba8a763cd9c is
//     verified at build time).
//   web/runner/data/m6e9-embed-tokens.json
//     The manifest: shape, byte counts, the raw BF16 source BLAKE3, and the
//     written f32 BLAKE3.
//   web/runner/data/m6e9-tokenizer.json
//     The byte-exact copy of the pinned tokenizer.json (11189060 bytes,
//     whole-file BLAKE3
//     664e6c2425fd92e710a67a919753493657005ddcd1cb839737b6678db3edf3c3,
//     verified at build time).
//
// Both artifacts are regenerable outputs (web/runner/data is git-ignored)
// and are verified after writing: re-read, exact byte counts and BLAKE3.
//
// Usage: node tools/build_m6e9_greedy_generation_artifacts.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { m6e6FastBlake3Hex } from "../web/tests/support/m6e6_decoder_stack_session_oracle.mjs";
import { callerOwnedBlake3Hex } from "../web/tests/support/m7c2b_qkv_source_oracle.mjs";
import {
  M6E7_OFFICIAL_PATHS,
  readOfficialBf16Tensor,
} from "../web/tests/support/m6e7_official_prefill_case.mjs";
import {
  M6E9_EMBED_RAW_BF16_BLAKE3,
  M6E9_TOKENIZER_BLAKE3,
  M6E9_TOKENIZER_BYTES,
  M6E9_VOCAB_SIZE,
} from "../web/tests/support/m6e9_greedy_generation_oracle.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HIDDEN = 1024;
const EMBED_OUT = path.join(ROOT, "web/runner/data/m6e9-embed-tokens.f32");
const EMBED_MANIFEST_OUT = path.join(ROOT, "web/runner/data/m6e9-embed-tokens.json");
const TOKENIZER_OUT = path.join(ROOT, "web/runner/data/m6e9-tokenizer.json");
const TOKENIZER_SOURCE = path.join(
  ROOT,
  "models/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e/tokenizer.json",
);

function fail(message) {
  console.error(`m6e9 artifact builder: ${message}`);
  process.exit(1);
}

console.log("reading model.embed_tokens.weight (positioned read of the pinned snapshot)…");
const embed = readOfficialBf16Tensor(
  ROOT,
  M6E7_OFFICIAL_PATHS.model,
  "model.embed_tokens.weight",
  [M6E9_VOCAB_SIZE, HIDDEN],
);
const embedRawBlake3 = m6e6FastBlake3Hex(embed.raw);
if (embedRawBlake3 !== M6E9_EMBED_RAW_BF16_BLAKE3) {
  fail(
    `raw BF16 embed_tokens.weight BLAKE3 drifted: ${embedRawBlake3} ` +
      `(expected ${M6E9_EMBED_RAW_BF16_BLAKE3})`,
  );
}
console.log(`raw BF16 embed_tokens BLAKE3 pinned: ${embedRawBlake3}`);
// The widened table is additionally scanned for finiteness at build time
// (readOfficialBf16Tensor already fails on a nonfinite BF16 element; the
// f32 table the browser driver consumes must carry the same guarantee).
for (let index = 0; index < embed.values.length; index += 1) {
  if (!Number.isFinite(embed.values[index])) {
    fail(`embed f32 table contains a nonfinite element at ${index}`);
  }
}

console.log("reading the pinned tokenizer.json…");
const tokenizerBytes = fs.readFileSync(TOKENIZER_SOURCE);
if (tokenizerBytes.byteLength !== M6E9_TOKENIZER_BYTES) {
  fail(`tokenizer.json length drifted: ${tokenizerBytes.byteLength}`);
}
const tokenizerBlake3 = callerOwnedBlake3Hex(tokenizerBytes);
if (tokenizerBlake3 !== M6E9_TOKENIZER_BLAKE3) {
  fail(
    `tokenizer.json BLAKE3 drifted: ${tokenizerBlake3} (expected ${M6E9_TOKENIZER_BLAKE3})`,
  );
}
console.log(`tokenizer.json BLAKE3 pinned: ${tokenizerBlake3}`);

const embedF32Bytes = new Uint8Array(
  embed.values.buffer,
  embed.values.byteOffset,
  embed.values.byteLength,
);
const embedF32Blake3 = m6e6FastBlake3Hex(embedF32Bytes);
const manifest = {
  schema_version: 1,
  tensor: "model.embed_tokens.weight",
  shape: [M6E9_VOCAB_SIZE, HIDDEN],
  dtype: "f32-le",
  bytes: embedF32Bytes.byteLength,
  f32_blake3: embedF32Blake3,
  source_bf16_blake3: embedRawBlake3,
  model_revision: "66317acc4c9fc17bd154591ce650735cd2855f3e",
};

fs.mkdirSync(path.dirname(EMBED_OUT), { recursive: true });
fs.writeFileSync(EMBED_OUT, embedF32Bytes);
fs.writeFileSync(EMBED_MANIFEST_OUT, `${JSON.stringify(manifest, null, 2)}\n`);
fs.writeFileSync(TOKENIZER_OUT, tokenizerBytes);

console.log("verifying the written artifacts…");
const writtenEmbed = fs.readFileSync(EMBED_OUT);
if (writtenEmbed.byteLength !== M6E9_VOCAB_SIZE * HIDDEN * 4) {
  fail("written embedding table length drifted");
}
if (m6e6FastBlake3Hex(writtenEmbed) !== embedF32Blake3) {
  fail("written embedding table BLAKE3 drifted");
}
const writtenManifest = JSON.parse(fs.readFileSync(EMBED_MANIFEST_OUT, "utf8"));
if (writtenManifest.f32_blake3 !== embedF32Blake3 ||
    writtenManifest.source_bf16_blake3 !== M6E9_EMBED_RAW_BF16_BLAKE3 ||
    JSON.stringify(writtenManifest.shape) !== JSON.stringify([M6E9_VOCAB_SIZE, HIDDEN])) {
  fail("written embedding manifest drifted");
}
const writtenTokenizer = fs.readFileSync(TOKENIZER_OUT);
if (writtenTokenizer.byteLength !== M6E9_TOKENIZER_BYTES ||
    callerOwnedBlake3Hex(writtenTokenizer) !== M6E9_TOKENIZER_BLAKE3) {
  fail("written tokenizer copy drifted");
}

console.log(JSON.stringify({
  status: "passed",
  embed: path.relative(ROOT, EMBED_OUT),
  embed_bytes: writtenEmbed.byteLength,
  embed_f32_blake3: embedF32Blake3,
  embed_source_bf16_blake3: embedRawBlake3,
  manifest: path.relative(ROOT, EMBED_MANIFEST_OUT),
  tokenizer: path.relative(ROOT, TOKENIZER_OUT),
  tokenizer_bytes: writtenTokenizer.byteLength,
  tokenizer_blake3: tokenizerBlake3,
}, null, 2));
