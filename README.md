# PaddleOCR-VL WebGPU

Try SotaOCR online: [https://sotaocr.com/en/free-ocr](https://sotaocr.com/en/free-ocr)

Run the PaddleOCR-VL 1.6 network directly in the browser with a
model-specific Rust/WASM/WebGPU inference engine.

This Apache-2.0 repository contains the inference essentials:

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
