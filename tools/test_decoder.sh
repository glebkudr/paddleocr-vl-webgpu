#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-fast}"

run_decoder_rust_contracts() {
  cargo test --jobs 1 \
    -p pvlc-runtime-core \
      --test decoder_gqa_split_plan_contract \
      --test gemv_tiled_plan_contract \
    -p pvlc-wgsl \
      --test decoder_gqa_split_wgsl_contract \
      --test gemv_tiled_wgsl_contract \
    -p pvlc-runtime-web \
      --test decoder_stack_split_gqa_web_module_contract \
      --test decoder_stack_tiled_gemv_web_module_contract \
    -- --test-threads=1
}

run_decoder_node_smoke() {
  node --test \
    --test-concurrency=1 \
    --test-name-pattern='plan pins|stack hook replaces' \
    web/tests/m7o2_split_gqa_contract.test.mjs \
    web/tests/m7o5_tiled_gemv_contract.test.mjs
}

run_decoder_node_numerical() {
  node --test \
    --test-concurrency=1 \
    web/tests/m7o2_split_gqa_contract.test.mjs \
    web/tests/m7o5_tiled_gemv_contract.test.mjs
}

run_frontend_full() {
  node --test --test-concurrency=1 web/tests/*.test.mjs
}

run_fp16_rust_contracts() {
  cargo test --jobs 1 \
    -p pvlc-safetensors \
      --test fp16_checkpoint_conversion_contract \
    -p pvlc-cli \
      --test fp16_checkpoint_cli_contract \
    -p pvlc-runtime-core \
      --test decoder_weight_storage_contract \
    -p pvlc-wgsl \
      --test decoder_fp16_weight_wgsl_contract \
    -p pvlc-runtime-web \
      --test decoder_stack_fp16_web_module_contract \
    -- --test-threads=1
}

run_fp16_python_contracts() {
  .venv/bin/python -m pytest -q \
    tools/reference_capture/tests/test_m7p_fp16_native_benchmark.py
}

run_fp16_node_contracts() {
  node --test \
    --test-concurrency=1 \
    web/tests/m7q1_fp16_decoder_contract.test.mjs \
    web/tests/m7q1_fp16_balanced_pack_contract.test.mjs \
    web/tests/m7q1_fp16_checkpoint_identity_contract.test.mjs
}

run_fp16_contracts() {
  run_fp16_rust_contracts
  run_fp16_python_contracts
  run_fp16_node_contracts
}

case "$mode" in
  fast)
    run_decoder_rust_contracts
    run_decoder_node_smoke
    ;;
  numerical)
    run_decoder_rust_contracts
    run_decoder_node_numerical
    ;;
  frontend)
    run_frontend_full
    ;;
  fp16)
    run_fp16_contracts
    ;;
  fp16-browser)
    run_fp16_contracts
    "$repo_root/web/tests/run_m7q1_fp16_decoder_browser.sh"
    ;;
  full)
    cargo test --jobs 1 \
      --workspace \
      --no-fail-fast \
      -- --test-threads=1
    .venv/bin/python -m pytest -q
    run_frontend_full
    ;;
  *)
    echo "usage: tools/test_decoder.sh {fast|numerical|fp16|fp16-browser|frontend|full}" >&2
    exit 2
    ;;
esac
