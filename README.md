# PaddleOCR-VL WebGPU

**The only browser-native port of PaddleOCR-VL 1.6—free, open source and
powered by WebGPU.**

Add client-side OCR for text, tables, formulas and charts to your web app with
no API key, hosted inference backend or per-page bill.

PaddleOCR-VL WebGPU runs locally in compatible browsers through a model-specific
Rust, WebAssembly and WebGPU engine. Pass it an image or prepared canvas, choose
an OCR task and receive raw model output through a focused JavaScript API.

**[Try this model online](https://sotaocr.com/en/free-ocr)**

## Why developers choose it

- **Free and open source:** Apache-2.0 code with no per-page OCR fees.
- **Browser-local inference:** the included example does not send image bytes
  to SotaOCR; public model files and static assets are downloaded from
  configured origins.
- **Concrete OCR tasks:** supports `ocr`, `table`, `formula` and `chart`.
- **Focused JavaScript API:** accepts an image or prepared canvas and returns
  raw model output.
- **Built for compatible GPUs:** specialized WebGPU kernels run in browsers
  with WebGPU and `shader-f16`.
- **Reproducible:** pinned model revisions, integrity checks and committed WASM
  artifacts.

## Measured browser performance vs. native

Same FP16 checkpoint, same table-recognition prompt, Apple M4 Pro, with warm
sequential inference:

| Measurement | Browser WebGPU | Native PyTorch/MPS | Relative result |
| --- | ---: | ---: | ---: |
| Vision encoder + projector latency | 0.638 s | 0.510 s | Browser 1.25× |
| Autoregressive decode throughput | 79.83 tok/s | 97.26 tok/s | Browser 1.22× higher per-token latency |
| End-to-end elapsed time | 17.152 s | 12.335 s | Browser 1.39× |

The timed end-to-end runs produced different output lengths—1,041 browser
tokens and 1,014 native tokens—so decode is compared using throughput and
per-token latency. Their first 256 generated tokens were byte-identical.

In a separate full-output parity run, not used for the timing table, browser
and native produced the same 1,014-token, 2,167-character result after removing
the browser-only `Table Recognition: ` UI prefix.

Model loading was excluded. This is a single-input benchmark, not a broad
accuracy or hardware study; performance varies by browser, GPU, and document.

This repository contains the inference essentials:

- the AOT compiler and model-pack tools;
- Rust/WASM runtime and specialized WGSL kernels;
- model-required image preprocessing, tokenizer and generation loop;
- pinned WebGPU model-pack loading and browser caching;
- a minimal single-image example returning raw model output;
- correctness contracts and benchmark tooling.

The full SotaOCR document pipeline — PDF rasterization, orientation,
unwarping, layout detection, crop routing, reading order and Markdown
assembly — lives in the separate `sotaocr-browser-ocr` repository under the
Attribution Link License 1.0.

## Quick start

Requirements:

- current Chrome or another browser with WebGPU and `shader-f16`;
- an HTTPS origin for production; localhost works during development;
- about 2.1 GB for the first model download and browser cache;
- 8 GB of system memory is recommended.

```bash
npm install
npm run dev
```

Choose one image and a PaddleOCR-VL task. The example returns the network's raw
text together with timing and token information.

## Package API

```js
import {
  createPaddleOcrVlEngine,
} from "@sotaocr/paddleocr-vl-webgpu";

const engine = await createPaddleOcrVlEngine();

try {
  const result = await engine.recognizeImage(file, {
    task: "ocr",
    maxGeneratedTokens: 512,
    onProgress(update) {
      console.log(update);
    },
  });
  console.log(result.text);
} finally {
  await engine.dispose();
}
```

For document-pipeline integrations, pass an already prepared canvas to
`recognizeCanvas()`. Supported task names are `ocr`, `table`, `formula` and
`chart`.

## Development

```bash
npm test
npm run build
cargo test --workspace
cargo check -p pvlc-runtime-web --target wasm32-unknown-unknown
```

The compiled WASM package is committed so the example works immediately.
Rebuild it with `npm run build:wasm`.

Model files are not committed. The browser downloads the pinned model packs
declared in `web/engine/browser_ocr_runtime.mjs`.

## Scope

The package deliberately stops at model inference. It does not contain:

- PDF parsing or rasterization;
- orientation, unwarping or layout ONNX models;
- page/block routing and reading-order reconstruction;
- full-document JSON or Markdown assembly;
- the SotaOCR product site, API, authentication or billing.

Keeping this boundary makes the inference engine independently reusable under
Apache-2.0.

## License

Code in this repository is licensed under the
[Apache License 2.0](LICENSE). Model weights and third-party dependencies
retain their respective licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Apache-2.0 does not grant rights to SotaOCR trademarks or logos beyond
reasonable description of the software's origin.
