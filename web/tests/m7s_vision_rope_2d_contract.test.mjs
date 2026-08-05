import assert from "node:assert/strict";
import test from "node:test";

import {
  MULTIMODAL,
  runSyntheticVisionStack,
  visionRope2dTables,
} from "../engine/multimodal_ocr.mjs";

function assertClose(actual, expected, tolerance = 2e-6) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `value ${index}: ${actual[index]} != ${expected[index]}`,
    );
  }
}

test("browser vision RoPE tables match Transformers H-then-W frequency layout", () => {
  const { cos, sin } = visionRope2dTables([1, 1, 2], {
    headDim: 8,
    theta: 10_000,
  });

  assert.equal(cos.length, 8);
  assert.equal(sin.length, 8);
  assert.deepEqual([...cos.subarray(0, 4)], [1, 1, 1, 1]);
  assert.deepEqual([...sin.subarray(0, 4)], [0, 0, 0, 0]);
  assertClose(cos.subarray(4), [1, 1, Math.cos(1), Math.cos(0.01)]);
  assertClose(sin.subarray(4), [0, 0, Math.sin(1), Math.sin(0.01)]);
});

test("browser vision RoPE repeats spatial rows for temporal frames and supports packed grids", () => {
  const { cos, sin } = visionRope2dTables([[2, 2, 1], [1, 1, 2]], {
    headDim: 8,
  });

  const identityCos = [1, 1, 1, 1];
  const identitySin = [0, 0, 0, 0];
  const heightCos = [Math.cos(1), Math.cos(0.01), 1, 1];
  const heightSin = [Math.sin(1), Math.sin(0.01), 0, 0];
  const widthCos = [1, 1, Math.cos(1), Math.cos(0.01)];
  const widthSin = [0, 0, Math.sin(1), Math.sin(0.01)];
  assertClose(cos, [
    ...identityCos,
    ...heightCos,
    ...identityCos,
    ...heightCos,
    ...identityCos,
    ...widthCos,
  ]);
  assertClose(sin, [
    ...identitySin,
    ...heightSin,
    ...identitySin,
    ...heightSin,
    ...identitySin,
    ...widthSin,
  ]);
});

test("browser vision RoPE rejects geometry that cannot match PaddleOCR-VL", () => {
  for (const [grid, options] of [
    [[1, 0, 2], { headDim: 8 }],
    [[1, 1, 2], { headDim: 6 }],
    [[1, 1, 2], { headDim: 8, theta: 0 }],
    [[], { headDim: 8 }],
  ]) {
    assert.throws(() => visionRope2dTables(grid, options), /vision RoPE/);
  }
});

test("OCR vision stack configures spatial RoPE before any GPU execution", async () => {
  const events = [];
  const tokens = 2;
  const planeBytes = tokens * MULTIMODAL.visionHidden * 4;
  const manifest = {
    tokens,
    layer_count: 2,
    checkpoint_layers: [],
    shards: [
      { id: "input.embeddings" },
      { id: "weights.vision_layer.00" },
      { id: "weights.vision_layer.01" },
      { id: "weights.vision_post_norm" },
    ],
  };
  const runtime = {
    compile_vision_encoder_stack_qkv_selection() {
      return {};
    },
    begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json() {
      events.push("begin");
      return '{"phase":"preflight"}';
    },
    configure_vision_encoder_stack_spatial_rope_f32(cos, sin) {
      events.push("rope");
      assert.ok(cos instanceof Float32Array);
      assert.ok(sin instanceof Float32Array);
      assert.equal(cos.length, tokens * 36);
      assert.equal(sin.length, tokens * 36);
    },
    preflight_vision_encoder_stack_manifest_shard_json(id) {
      events.push(`preflight:${id}`);
    },
    async start_vision_encoder_stack_sharded_json() {
      events.push("start");
    },
    enqueue_vision_encoder_stack_sharded_layer_json(id) {
      events.push(`layer:${id}`);
      return "{}";
    },
    async finish_vision_encoder_stack_sharded() {
      events.push("finish");
      return {
        checkpoint_bytes: new Uint8Array(planeBytes),
        diagnostics_json: "{}",
      };
    },
  };

  await runSyntheticVisionStack(
    runtime,
    manifest,
    async () => new Uint8Array(),
    new Uint8Array(planeBytes),
    { gridThw: [1, 1, 2] },
  );

  assert.deepEqual(events, [
    "begin",
    "rope",
    "preflight:input.embeddings",
    "preflight:weights.vision_layer.00",
    "preflight:weights.vision_layer.01",
    "preflight:weights.vision_post_norm",
    "start",
    "layer:weights.vision_layer.00",
    "layer:weights.vision_layer.01",
    "finish",
  ]);
});
