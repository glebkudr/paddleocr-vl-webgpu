import assert from "node:assert/strict";
import test from "node:test";

import {
  diagnoseBrowser,
  MODEL_DOWNLOAD_BYTES,
} from "../browser_diagnostics.mjs";

test("diagnostics reject browsers without WebGPU", async () => {
  const result = await diagnoseBrowser({}, {});

  assert.equal(result.status, "unsupported");
  assert.deepEqual(result.reasons, ["webgpu_unavailable"]);
});

test("diagnostics require shader-f16 and sufficient GPU buffer limits", async () => {
  const result = await diagnoseBrowser({
    gpu: {
      async requestAdapter() {
        return {
          features: { has: () => false },
          limits: {
            maxBufferSize: 64 * 1024 * 1024,
            maxStorageBufferBindingSize: 64 * 1024 * 1024,
          },
        };
      },
    },
    deviceMemory: 16,
    storage: {
      async estimate() {
        return { quota: MODEL_DOWNLOAD_BYTES * 3, usage: 0 };
      },
    },
  }, {
    memory: { jsHeapSizeLimit: MODEL_DOWNLOAD_BYTES * 3 },
  });

  assert.equal(result.status, "unsupported");
  assert.ok(result.reasons.includes("shader_f16_unavailable"));
  assert.ok(result.reasons.includes("gpu_buffer_limits_low"));
});

test("diagnostics accept a capable browser", async () => {
  const result = await diagnoseBrowser({
    gpu: {
      async requestAdapter() {
        return {
          features: { has: (feature) => feature === "shader-f16" },
          limits: {
            maxBufferSize: 512 * 1024 * 1024,
            maxStorageBufferBindingSize: 256 * 1024 * 1024,
          },
        };
      },
    },
    deviceMemory: 16,
    storage: {
      async estimate() {
        return { quota: 10_000_000_000, usage: 1_000_000_000 };
      },
    },
  }, {
    memory: { jsHeapSizeLimit: 4_000_000_000 },
  });

  assert.equal(result.status, "ready");
  assert.equal(result.memory, "enough");
  assert.deepEqual(result.reasons, []);
});
