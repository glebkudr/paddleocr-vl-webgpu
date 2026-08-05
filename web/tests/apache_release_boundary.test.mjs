import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "../..");
const read = (relativePath) =>
  fs.readFileSync(path.join(root, relativePath), "utf8");

test("the essentials package is unambiguously Apache-2.0", () => {
  const packageJson = JSON.parse(read("package.json"));
  const cargo = read("Cargo.toml");
  const license = read("LICENSE");

  assert.equal(packageJson.name, "@sotaocr/paddleocr-vl-webgpu");
  assert.equal(packageJson.license, "Apache-2.0");
  assert.equal(packageJson.exports["."], "./web/engine/browser_ocr_runtime.mjs");
  assert.match(cargo, /license = "Apache-2\.0"/);
  assert.doesNotMatch(cargo, /license-file/);
  assert.match(license, /^\s*Apache License\s+Version 2\.0, January 2004/);
  assert.doesNotMatch(license, /ATTRIBUTION LINK LICENSE/);
});

test("the network runtime is independent from the document pipeline", () => {
  const runtime = read("web/engine/browser_ocr_runtime.mjs");

  assert.match(runtime, /export class PaddleOcrVlEngine/);
  assert.match(runtime, /export async function createPaddleOcrVlEngine/);
  assert.doesNotMatch(runtime, /BrowserDocumentPipeline/);
  assert.doesNotMatch(runtime, /document_pipeline/);
  assert.doesNotMatch(runtime, /browser_onnx_runtime/);

  for (const forbidden of [
    "web/document_ocr.mjs",
    "web/engine/document_pipeline.mjs",
    "web/engine/browser_onnx_runtime.mjs",
    "web/engine/sota_document_contract.mjs",
    "web/engine/otsl_to_html.mjs",
  ]) {
    assert.equal(fs.existsSync(path.join(root, forbidden)), false, forbidden);
  }
});

test("the Apache repository includes only a minimal single-image example", () => {
  assert.equal(fs.existsSync(path.join(root, "examples/basic/index.html")), true);
  assert.equal(fs.existsSync(path.join(root, "examples/basic/app.mjs")), true);
  assert.equal(fs.existsSync(path.join(root, "web/app.mjs")), false);
  assert.equal(fs.existsSync(path.join(root, "web/index.html")), false);
});
