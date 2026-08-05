import assert from "node:assert/strict";
import test from "node:test";

import { runGreedyGeneration } from "../engine/greedy_generation.mjs";
import {
  AdaptiveOcrConcurrency,
  BrowserRuntimePool,
  recommendOcrParallelism,
} from "../engine/ocr_concurrency.mjs";

test("RTX 3090 starts with a single warm-up lane and admits four lanes", () => {
  const profile = recommendOcrParallelism(
    {
      adapter_name: "NVIDIA GeForce RTX 3090",
      adapter_device_type: "discretegpu",
    },
    { hardwareConcurrency: 24, deviceMemory: 8 },
  );
  assert.equal(profile.initialParallelism, 1);
  assert.equal(profile.maxParallelism, 4);
  assert.equal(profile.reason, "large_discrete_gpu");
});

test("adaptive concurrency warms alone, ramps on throughput and backs off", () => {
  const controller = new AdaptiveOcrConcurrency({
    maxParallelism: 4,
    initialParallelism: 1,
    reason: "fixture",
  });
  assert.equal(controller.current(30), 1);
  controller.recordWave({
    width: 1,
    wallMs: 100,
    outcomes: [{ ok: true, result: { generatedTokens: 100 } }],
  });
  assert.equal(controller.current(29), 2);
  controller.recordWave({
    width: 2,
    wallMs: 100,
    outcomes: [
      { ok: true, result: { generatedTokens: 100 } },
      { ok: true, result: { generatedTokens: 100 } },
    ],
  });
  assert.equal(controller.current(27), 3);
  controller.recordWave({
    width: 3,
    wallMs: 500,
    outcomes: [
      { ok: true, result: { generatedTokens: 20 } },
      { ok: true, result: { generatedTokens: 20 } },
      { ok: true, result: { generatedTokens: 20 } },
    ],
  });
  assert.equal(controller.current(24), 2);
});

test("runtime pool never lends one runtime to two simultaneous jobs", async () => {
  const runtimes = [{ id: 1, free() {} }, { id: 2, free() {} }];
  const pool = new BrowserRuntimePool(runtimes);
  const active = new Set();
  let overlap = 0;
  const run = (delay) => pool.run(async (runtime) => {
    assert.equal(active.has(runtime.id), false);
    active.add(runtime.id);
    overlap = Math.max(overlap, active.size);
    await new Promise((resolve) => setTimeout(resolve, delay));
    active.delete(runtime.id);
  });
  await Promise.all([run(15), run(15), run(1)]);
  assert.equal(overlap, 2);
  pool.dispose();
});

test("greedy generation uses GPU top-1 and reads eight bytes per token", async () => {
  const selected = [7, 2];
  let cacheTokens = 3;
  let logitsCalls = 0;
  const runtime = {
    async top1_decoder_stack_session() {
      const tokenId = selected.shift();
      return {
        token_id: tokenId,
        value: 10,
        diagnostics_json: JSON.stringify({
          cache_tokens: cacheTokens,
          readback_bytes: 8,
          queue_wall_time_ns: 12,
        }),
      };
    },
    async logits_decoder_stack_session() {
      logitsCalls += 1;
      throw new Error("full logits readback must be elided");
    },
    async step_decoder_stack_session() {
      cacheTokens += 1;
      return {
        diagnostics_json: JSON.stringify({
          cache_tokens_after: cacheTokens,
        }),
      };
    },
  };
  const result = await runGreedyGeneration(runtime, {
    embedding: { gather: () => new Float32Array(1024) },
    eosTokenId: 2,
    maxSteps: 8,
    cacheCapacity: 32,
    initialCacheTokens: 3,
  });
  assert.deepEqual(result.tokenIds, [7, 2]);
  assert.equal(result.stopReason, "eos");
  assert.equal(result.gpuTop1, true);
  assert.equal(result.logitsReadbackBytes, 16);
  assert.equal(result.selectionQueueWallTimeNs, 24);
  assert.equal(logitsCalls, 0);
});
