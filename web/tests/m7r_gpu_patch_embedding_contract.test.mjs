import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  MULTIMODAL,
  buildMultimodalSessionPack,
  enqueueVisionStackLayer,
  patchEmbeddingWebGpu,
  preflightVisionStackShards,
  readSessionPackShardBytes,
  readSessionPackShards,
  runSyntheticVisionStack,
} from "../engine/multimodal_ocr.mjs";
import { callerOwnedBlake3Hex } from "./support/m7c2b_qkv_source_oracle.mjs";

function f32Bytes(values) {
  const copy = new Float32Array(values);
  return new Uint8Array(copy.buffer);
}

test("browser patch embedding sends exact F32 operands to the WebGPU bytes API", async () => {
  const gridThw = [1, 2, 2];
  const patchCount = 4;
  const inputWidth = 3 * MULTIMODAL.patchSize * MULTIMODAL.patchSize;
  const hiddenSize = MULTIMODAL.visionHidden;
  const pixelValues = new Float32Array(patchCount * inputWidth);
  pixelValues[0] = 1.25;
  pixelValues[pixelValues.length - 1] = -2.5;
  const weight = new Float32Array(hiddenSize * inputWidth);
  weight[0] = 0.75;
  weight[weight.length - 1] = -0.5;
  const bias = new Float32Array(hiddenSize);
  bias[0] = 0.125;
  bias[bias.length - 1] = -0.25;
  const positionEmbedding = new Float32Array(
    MULTIMODAL.positionGrid * MULTIMODAL.positionGrid * hiddenSize,
  );

  const projected = new Float32Array(patchCount * hiddenSize);
  for (let patch = 0; patch < patchCount; patch += 1) {
    projected[patch * hiddenSize] = patch + 10;
    projected[(patch + 1) * hiddenSize - 1] = -(patch + 20);
  }
  let calls = 0;
  const runtime = {
    async run_vision_patch_projection_bytes(
      descriptorJson,
      inputBytes,
      weightBytes,
      biasBytes,
    ) {
      calls += 1;
      assert.deepEqual(JSON.parse(descriptorJson), {
        schema_version: 1,
        patch_count: patchCount,
        input_width: inputWidth,
        output_width: hiddenSize,
        weight_storage: "f32",
      });
      assert.deepEqual(inputBytes, f32Bytes(pixelValues));
      assert.deepEqual(weightBytes, f32Bytes(weight));
      assert.deepEqual(biasBytes, f32Bytes(bias));
      return {
        checkpoint_bytes: f32Bytes(projected),
        diagnostics_json: JSON.stringify({
          kernel: "vision_patch_projection_f32",
          queue_wall_time_ns: 123_000,
        }),
      };
    },
  };

  const result = await patchEmbeddingWebGpu(runtime, pixelValues, {
    weight,
    bias,
    positionEmbedding,
    gridThw,
  });

  assert.equal(calls, 1);
  assert.deepEqual(result.patch, projected);
  assert.deepEqual(result.output, projected);
  assert.deepEqual(result.diagnostics, {
    kernel: "vision_patch_projection_f32",
    queue_wall_time_ns: 123_000,
  });
});

test("browser patch embedding rejects malformed runtime output", async () => {
  const gridThw = [1, 2, 2];
  const patchCount = 4;
  const inputWidth = 3 * MULTIMODAL.patchSize * MULTIMODAL.patchSize;
  const hiddenSize = MULTIMODAL.visionHidden;
  const operands = {
    weight: new Float32Array(hiddenSize * inputWidth),
    bias: new Float32Array(hiddenSize),
    positionEmbedding: new Float32Array(
      MULTIMODAL.positionGrid * MULTIMODAL.positionGrid * hiddenSize,
    ),
    gridThw,
  };
  const pixelValues = new Float32Array(patchCount * inputWidth);

  await assert.rejects(
    patchEmbeddingWebGpu(
      {
        async run_vision_patch_projection_bytes() {
          return {
            checkpoint_bytes: new Uint8Array(patchCount * hiddenSize * 4 - 4),
            diagnostics_json: "{}",
          };
        },
      },
      pixelValues,
      operands,
    ),
    /output byte length drifted/,
  );
  await assert.rejects(
    patchEmbeddingWebGpu({}, pixelValues, operands),
    /runtime bytes API is unavailable/,
  );
});

test("OCR vision loop prefers synchronous streaming enqueue without creating an awaitable", async () => {
  let legacyCalls = 0;
  const payload = new Uint8Array([1, 2, 3]);
  const submitted = enqueueVisionStackLayer(
    {
      enqueue_vision_encoder_stack_sharded_layer_json(id, bytes) {
        assert.equal(id, "vision.layer.07");
        assert.equal(bytes, payload);
        return '{"phase":"layers","next_layer":8}';
      },
      async run_vision_encoder_stack_sharded_layer_json() {
        legacyCalls += 1;
      },
    },
    "vision.layer.07",
    payload,
  );

  assert.deepEqual(submitted, {
    streaming: true,
    status: '{"phase":"layers","next_layer":8}',
  });
  assert.equal(legacyCalls, 0);
  assert.equal("completion" in submitted, false);
});

test("OCR vision loop retains the accepted awaited layer API as fallback", async () => {
  const payload = new Uint8Array([4, 5, 6]);
  let calls = 0;
  const submitted = enqueueVisionStackLayer(
    {
      async run_vision_encoder_stack_sharded_layer_json(id, bytes) {
        calls += 1;
        assert.equal(id, "vision.layer.12");
        assert.equal(bytes, payload);
        return '{"phase":"layers","next_layer":13}';
      },
    },
    "vision.layer.12",
    payload,
  );

  assert.equal(submitted.streaming, false);
  assert.equal(await submitted.completion, '{"phase":"layers","next_layer":13}');
  assert.equal(calls, 1);
  assert.throws(
    () => enqueueVisionStackLayer({}, "vision.layer.12", payload),
    /layer API is unavailable/,
  );
});

test("OCR vision preflight prefers manifest declarations without loading 1.65 GB twice", async () => {
  const shards = [
    { id: "input.embeddings" },
    { id: "weights.vision_layer.00" },
    { id: "weights.vision_post_norm" },
  ];
  const accepted = [];
  let payloadLoads = 0;
  const mode = await preflightVisionStackShards(
    {
      preflight_vision_encoder_stack_manifest_shard_json(...args) {
        assert.equal(args.length, 1);
        const [id] = args;
        accepted.push(id);
      },
    },
    shards,
    async () => {
      payloadLoads += 1;
      throw new Error("manifest preflight loaded a payload");
    },
    new Uint8Array([9]),
  );

  assert.equal(mode, "manifest");
  assert.deepEqual(accepted, shards.map(({ id }) => id));
  assert.equal(payloadLoads, 0);
});

test("OCR vision preflight retains exact payload-validation fallback", async () => {
  const shards = [
    { id: "input.embeddings" },
    { id: "weights.vision_layer.00" },
    { id: "weights.vision_post_norm" },
  ];
  const input = new Uint8Array([1]);
  const payloads = new Map([
    ["weights.vision_layer.00", new Uint8Array([2])],
    ["weights.vision_post_norm", new Uint8Array([3])],
  ]);
  const accepted = [];
  const mode = await preflightVisionStackShards(
    {
      preflight_vision_encoder_stack_shard_json(id, bytes) {
        accepted.push([id, bytes]);
      },
    },
    shards,
    async (id) => payloads.get(id),
    input,
  );

  assert.equal(mode, "payload");
  assert.deepEqual(accepted, [
    ["input.embeddings", input],
    ["weights.vision_layer.00", payloads.get("weights.vision_layer.00")],
    ["weights.vision_post_norm", payloads.get("weights.vision_post_norm")],
  ]);
});

test("real OCR vision sequence loads every weight shard exactly once when manifest preflight is available", async () => {
  const planeBytes = 2 * 1152 * 4;
  const manifest = {
    tokens: 2,
    layer_count: 2,
    checkpoint_layers: [],
    shards: [
      { id: "input.embeddings" },
      { id: "weights.vision_layer.00" },
      { id: "weights.vision_layer.01" },
      { id: "weights.vision_post_norm" },
    ],
  };
  const input = new Uint8Array(planeBytes);
  const loadCounts = new Map();
  const calls = [];
  const runtime = {
    compile_vision_encoder_stack_qkv_selection() {
      return {};
    },
    begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json() {
      return JSON.stringify({ phase: "preflight" });
    },
    preflight_vision_encoder_stack_manifest_shard_json(...args) {
      assert.equal(args.length, 1);
      const [id] = args;
      calls.push(["preflight", id]);
    },
    async start_vision_encoder_stack_sharded_json(id, bytes) {
      calls.push(["start", id, bytes]);
    },
    enqueue_vision_encoder_stack_sharded_layer_json(id, bytes) {
      calls.push(["layer", id, bytes]);
      return "{}";
    },
    async finish_vision_encoder_stack_sharded(id, bytes) {
      calls.push(["finish", id, bytes]);
      return {
        checkpoint_bytes: new Uint8Array(planeBytes),
        diagnostics_json: JSON.stringify({}),
      };
    },
  };
  const result = await runSyntheticVisionStack(
    runtime,
    manifest,
    async (id) => {
      loadCounts.set(id, (loadCounts.get(id) ?? 0) + 1);
      return new Uint8Array([loadCounts.size]);
    },
    input,
  );

  assert.equal(result.bytes.byteLength, planeBytes);
  assert.deepEqual(
    [...loadCounts],
    [
      ["weights.vision_layer.00", 1],
      ["weights.vision_layer.01", 1],
      ["weights.vision_post_norm", 1],
    ],
    "the OCR path reintroduced a duplicate preflight payload read",
  );
  assert.deepEqual(
    calls.filter(([kind]) => kind === "preflight").map(([, id]) => id),
    manifest.shards.map(({ id }) => id),
  );
  assert.equal(calls.filter(([kind]) => kind === "layer").length, 2);
});

test("public browser API is bound to the shared tested vision-stack runner", async () => {
  const demoSource = await readFile(
    new URL("../engine/browser_ocr_runtime.mjs", import.meta.url),
    "utf8",
  );
  assert.match(
    demoSource,
    /import\s*\{[\s\S]*\brunSyntheticVisionStack\b[\s\S]*\}\s*from\s*["']\.\/multimodal_ocr\.mjs["']/,
    "browser_ocr_runtime must import the shared runner exercised above",
  );
  assert.doesNotMatch(
    demoSource,
    /(?:async\s+)?function\s+runSyntheticVisionStack\s*\(/,
    "browser_ocr_runtime must not retain an untested duplicate runner",
  );
  assert.match(
    demoSource,
    /m7q1-decoder-stack-logits-fp16-official\.pvlc/,
    "public OCR must use the pinned FP16 decoder source pack",
  );
  assert.match(demoSource, /\breadSessionPackShardBytes\b/);
  assert.match(demoSource, /weightStorage:\s*"f16"/);
});

test("multimodal session builder preserves F16 weights and dynamic F32 rope tables", () => {
  const f16 = (seed) => Uint8Array.of(seed, 0x3c, seed + 1, 0xbc);
  const weights = {
    norm1: f16(1),
    q: f16(3),
    k: f16(5),
    v: f16(7),
    o: f16(9),
    ropeCos: Float32Array.of(1, 0.5, -0.25),
    ropeSin: Float32Array.of(0, -0.5, 0.25),
    norm2: f16(11),
    gate: f16(13),
    up: f16(15),
    down: f16(17),
    finalNorm: f16(19),
    lmHead: f16(21),
  };
  const checkpointIdentity = {
    blake3: "7eaf17cbadb7ff816491a3bfe8c96abc52c85ceea5506e73f3eb676cff57655a",
    bytes: 1_917_255_480,
  };
  const pack = buildMultimodalSessionPack({
    descriptor: { prefix_tokens: 0, cache_capacity: 1_371 },
    weights,
    blake3Hex: callerOwnedBlake3Hex,
    oracle: "synthetic",
    caseId: "multimodal.fp16_contract",
    weightStorage: "f16",
    checkpointIdentity,
  });
  const raw = readSessionPackShardBytes(pack, callerOwnedBlake3Hex);
  const descriptor = JSON.parse(
    new TextDecoder().decode(raw.get("ir.decoder_stack_00")),
  );
  const view = new DataView(pack.buffer, pack.byteOffset, pack.byteLength);
  const manifestLength = view.getUint32(12, true);
  const manifest = JSON.parse(
    new TextDecoder().decode(pack.subarray(32, 32 + manifestLength)),
  );

  assert.equal(manifest.precision_profile, "balanced");
  assert.equal(descriptor.weight_storage, "f16");
  assert.equal(descriptor.checkpoint_blake3, checkpointIdentity.blake3);
  assert.equal(descriptor.checkpoint_bytes, checkpointIdentity.bytes);
  assert.equal(descriptor.shards["weights.q_proj"].dtype, "f16");
  assert.equal(descriptor.shards["weights.mrope_cos"].dtype, "f32");
  assert.deepEqual(raw.get("weights.q_proj"), weights.q);
  assert.deepEqual(
    raw.get("weights.mrope_cos"),
    new Uint8Array(
      weights.ropeCos.buffer,
      weights.ropeCos.byteOffset,
      weights.ropeCos.byteLength,
    ),
  );
});

test("multimodal session builder retains the accepted default F32 pack path", () => {
  const values = (seed) => Float32Array.of(seed, -seed, seed / 2);
  const weights = {
    norm1: values(1),
    q: values(2),
    k: values(3),
    v: values(4),
    o: values(5),
    ropeCos: values(6),
    ropeSin: values(7),
    norm2: values(8),
    gate: values(9),
    up: values(10),
    down: values(11),
    finalNorm: values(12),
    lmHead: values(13),
  };
  const pack = buildMultimodalSessionPack({
    descriptor: { prefix_tokens: 0, cache_capacity: 337 },
    weights,
    blake3Hex: callerOwnedBlake3Hex,
  });
  const shards = readSessionPackShards(pack, callerOwnedBlake3Hex);
  const raw = readSessionPackShardBytes(pack, callerOwnedBlake3Hex);
  const descriptor = JSON.parse(
    new TextDecoder().decode(raw.get("ir.decoder_stack_00")),
  );
  const manifestLength = new DataView(
    pack.buffer,
    pack.byteOffset,
    pack.byteLength,
  ).getUint32(12, true);
  const manifest = JSON.parse(
    new TextDecoder().decode(pack.subarray(32, 32 + manifestLength)),
  );

  assert.equal(manifest.precision_profile, "fidelity");
  assert.equal("weight_storage" in descriptor, false);
  assert.equal("dtype" in descriptor.shards["weights.q_proj"], false);
  assert.deepEqual(shards.get("weights.q_proj"), weights.q);
});

test("manifest preflight propagates its first error without payload or legacy fallback", async () => {
  const calls = [];
  let legacyCalls = 0;
  let payloadLoads = 0;
  await assert.rejects(
    preflightVisionStackShards(
      {
        preflight_vision_encoder_stack_manifest_shard_json(...args) {
          assert.equal(args.length, 1);
          const [id] = args;
          calls.push(id);
          if (id === "weights.vision_layer.00") {
            throw new Error("wrong shard order");
          }
        },
        preflight_vision_encoder_stack_shard_json() {
          legacyCalls += 1;
        },
      },
      [
        { id: "input.embeddings" },
        { id: "weights.vision_layer.00" },
        { id: "weights.vision_post_norm" },
      ],
      async () => {
        payloadLoads += 1;
        return new Uint8Array();
      },
      new Uint8Array([1]),
    ),
    /wrong shard order/,
  );
  assert.deepEqual(calls, [
    "input.embeddings",
    "weights.vision_layer.00",
  ]);
  assert.equal(legacyCalls, 0);
  assert.equal(payloadLoads, 0);
});
