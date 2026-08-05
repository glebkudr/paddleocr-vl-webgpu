# Third-party notices

## Model

| Component | Source | License |
|---|---|---|
| PaddleOCR-VL 1.6 WebGPU deployment artifacts | [glebkudr/PaddleOCR-VL-1.6-WebGPU](https://huggingface.co/glebkudr/PaddleOCR-VL-1.6-WebGPU) | Apache License 2.0 |
| Upstream PaddleOCR-VL 1.6 | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | Apache License 2.0 |

The deployment artifacts are deterministic format conversions of the pinned
upstream checkpoint. Selected tensors are converted to FP16/F32, transposed
for WebGPU kernels and packed into the engine container format. The model is
not fine-tuned.

## Browser and Rust dependencies

The browser package and Rust workspace use dependencies recorded in
`package-lock.json` and `Cargo.lock`. Those dependencies retain their own
licenses. Generate a dependency-license report before publishing binary
releases.
