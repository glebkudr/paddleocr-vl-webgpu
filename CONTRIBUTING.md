# Contributing

Thank you for improving PaddleOCR-VL WebGPU.

1. Open an issue before a large architectural change.
2. Keep the network runtime independent from document-level preprocessing and
   the private SotaOCR application.
3. Add or update focused tests for real user-facing behavior.
4. Run `npm test`, `npm run build`, and `cargo test --workspace`.
5. Preserve pinned model revisions and integrity checks.

Unless stated otherwise in writing when submitted, contributions are provided
under the Apache License 2.0, as described in `LICENSE`.
