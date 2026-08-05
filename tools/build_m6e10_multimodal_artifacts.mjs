#!/usr/bin/env node
// Builds the M6e10 multimodal artifacts:
//
//   web/runner/data/m6e10-vision-embed.safetensors
//     The exact BF16 -> f32 widening of the pinned checkpoint vision
//     embedding tensors:
//       visual.vision_model.embeddings.patch_embedding.weight [1152, 3, 14, 14]
//       visual.vision_model.embeddings.patch_embedding.bias   [1152]
//       visual.vision_model.embeddings.position_embedding.weight [729, 1152]
//     (all from models/snapshots/66317acc…/model.safetensors, read with
//     positioned reads; the 8-byte-aligned safetensors header keeps the
//     tensor payloads zero-copy viewable).
//   web/runner/data/m6e10/{clean_latin,table}/…
//     Byte-exact copies of the locked golden-bundle anchors the browser gate
//     consumes (the bundles live in artifacts/goldens, outside the web/
//     server root; every copied file is re-hashed after writing):
//       clean_latin: source-image.bin, processor.safetensors,
//         stage-checkpoints.safetensors, deep-checkpoints.safetensors
//       table:       source-image.bin, processor.safetensors,
//         stage-checkpoints.safetensors
//
// The artifacts are regenerable outputs (web/runner/data is git-ignored) and
// are verified after writing: re-read, exact shapes, finiteness, and BLAKE3
// digests in the output.
//
// Usage: node tools/build_m6e10_multimodal_artifacts.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { callerOwnedBlake3Hex } from "../web/tests/support/m7c2b_qkv_source_oracle.mjs";
import {
  M6E7_OFFICIAL_PATHS,
  readOfficialBf16Tensor,
} from "../web/tests/support/m6e7_official_prefill_case.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "web/runner/data/m6e10-vision-embed.safetensors");
const MODEL = M6E7_OFFICIAL_PATHS.model;

function fail(message) {
  console.error(`m6e10 artifact builder: ${message}`);
  process.exit(1);
}

const TENSORS = [
  ["patch_embedding_weight", "visual.vision_model.embeddings.patch_embedding.weight", [1152, 3, 14, 14]],
  ["patch_embedding_bias", "visual.vision_model.embeddings.patch_embedding.bias", [1152]],
  ["position_embedding_weight", "visual.vision_model.embeddings.position_embedding.weight", [729, 1152]],
];

console.log("reading the vision embedding tensors (positioned reads)…");
const tensors = TENSORS.map(([name, tensor, shape]) => {
  const { values } = readOfficialBf16Tensor(ROOT, MODEL, tensor, shape);
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      fail(`${tensor} contains a nonfinite F32 at ${index}`);
    }
  }
  return { name, shape, values };
});

function writeSafetensors(entries) {
  const header = {};
  let offset = 0;
  const parts = [];
  for (const { name, shape, values } of entries) {
    const bytes = values.byteLength;
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + bytes] };
    parts.push(new Uint8Array(values.buffer, values.byteOffset, bytes));
    offset += bytes;
  }
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

const bundle = writeSafetensors(tensors);
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, bundle);

// Byte-exact copies of the locked golden-bundle anchors the browser gate
// consumes (the bundles live in artifacts/goldens, outside the web/ server
// root). Every copied file is re-hashed after writing.
const CASE_ANCHORS = [
  {
    caseId: "clean_latin",
    sourceDir: path.join(ROOT, "artifacts/goldens/ocr.clean_latin.0001-l3"),
    anchors: [
      { name: "source-image.bin" },
      { name: "processor.safetensors" },
      { name: "stage-checkpoints.safetensors" },
      { name: "deep-checkpoints.safetensors" },
    ],
  },
  {
    caseId: "table",
    sourceDir: path.join(ROOT, "artifacts/goldens/table.simple.0001-l2"),
    anchors: [
      { name: "source-image.bin" },
      { name: "processor.safetensors" },
      { name: "stage-checkpoints.safetensors" },
    ],
  },
];
console.log("copying the golden-bundle anchors into web/runner/data/m6e10…");
for (const { caseId, sourceDir, anchors } of CASE_ANCHORS) {
  const outDir = path.join(ROOT, "web/runner/data/m6e10", caseId);
  fs.mkdirSync(outDir, { recursive: true });
  for (const { name } of anchors) {
    const source = path.join(sourceDir, name);
    if (!fs.existsSync(source)) {
      fail(`golden anchor is missing at ${source}`);
    }
    const bytes = fs.readFileSync(source);
    fs.writeFileSync(path.join(outDir, name), bytes);
    const writtenBack = fs.readFileSync(path.join(outDir, name));
    if (writtenBack.byteLength !== bytes.byteLength ||
        callerOwnedBlake3Hex(writtenBack) !== callerOwnedBlake3Hex(bytes)) {
      fail(`copied anchor ${caseId}/${name} is not byte-exact`);
    }
  }
}

console.log("verifying the written artifact…");
const written = fs.readFileSync(OUT);
const headerLength = Number(written.readBigUInt64LE(0));
const header = JSON.parse(written.subarray(8, 8 + headerLength).toString("utf8"));
for (const { name, shape } of tensors) {
  const info = header[name];
  if (info === undefined || info.dtype !== "F32" ||
      JSON.stringify(info.shape) !== JSON.stringify(shape)) {
    fail(`written bundle tensor ${name} drifted`);
  }
  const [begin, end] = info.data_offsets;
  const floats = new Float32Array(
    written.buffer,
    written.byteOffset + 8 + headerLength + begin,
    (end - begin) / 4,
  );
  for (let index = 0; index < floats.length; index += 1) {
    if (!Number.isFinite(floats[index])) {
      fail(`written bundle tensor ${name} contains a nonfinite F32 at ${index}`);
    }
  }
}

console.log(JSON.stringify({
  status: "passed",
  bundle: path.relative(ROOT, OUT),
  bundle_bytes: written.byteLength,
  bundle_blake3: callerOwnedBlake3Hex(new Uint8Array(written)),
  tensors: TENSORS.map(([name]) => name),
  case_anchors: CASE_ANCHORS.map(({ caseId, anchors }) => ({
    case: caseId,
    files: anchors.map(({ name }) => name),
  })),
}, null, 2));
