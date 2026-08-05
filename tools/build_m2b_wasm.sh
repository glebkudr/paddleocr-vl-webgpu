#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_WASM_BINDGEN="wasm-bindgen 0.2.126"
ACTUAL_WASM_BINDGEN="$(wasm-bindgen --version)"

if [[ "$ACTUAL_WASM_BINDGEN" != "$EXPECTED_WASM_BINDGEN" ]]; then
  echo "expected $EXPECTED_WASM_BINDGEN, got $ACTUAL_WASM_BINDGEN" >&2
  exit 1
fi

cargo build \
  --manifest-path "$ROOT/Cargo.toml" \
  -p pvlc-runtime-web \
  --target wasm32-unknown-unknown \
  --release
mkdir -p "$ROOT/web/runner/pkg"
wasm-bindgen \
  "$ROOT/target/wasm32-unknown-unknown/release/pvlc_runtime_web.wasm" \
  --target web \
  --out-dir "$ROOT/web/runner/pkg" \
  --out-name pvlc_runtime_web \
  --no-typescript
