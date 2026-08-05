export const MODEL_DOWNLOAD_BYTES = 2_038_286_862;
export const RECOMMENDED_DEVICE_MEMORY_GB = 8;
export const MIN_STORAGE_BYTES = 2_500_000_000;
export const MIN_BUFFER_BYTES = 256 * 1024 * 1024;
export const MIN_STORAGE_BUFFER_BYTES = 128 * 1024 * 1024;

function finitePositive(value) {
  const normalized = Number(value);
  return Number.isFinite(normalized) && normalized > 0 ? normalized : null;
}

async function storageAvailable(navigatorLike) {
  try {
    const estimate = await navigatorLike.storage?.estimate();
    const quota = finitePositive(estimate?.quota);
    const usage = Number(estimate?.usage);
    if (quota === null || !Number.isFinite(usage) || usage < 0) return null;
    return Math.max(0, quota - usage);
  } catch {
    return null;
  }
}

export async function diagnoseBrowser(
  navigatorLike = globalThis.navigator ?? {},
  performanceLike = globalThis.performance ?? {},
) {
  const deviceMemoryGB = finitePositive(navigatorLike.deviceMemory);
  const jsHeapLimitBytes = finitePositive(
    performanceLike.memory?.jsHeapSizeLimit,
  );
  const storageAvailableBytes = await storageAvailable(navigatorLike);

  if (!navigatorLike.gpu) {
    return {
      status: "unsupported",
      memory: deviceMemoryGB === null ? "unknown" : "limited",
      webgpu: false,
      adapter: false,
      shaderF16: false,
      deviceMemoryGB,
      jsHeapLimitBytes,
      storageAvailableBytes,
      maxBufferSize: null,
      maxStorageBufferBindingSize: null,
      reasons: ["webgpu_unavailable"],
    };
  }

  let adapter = null;
  try {
    adapter = await navigatorLike.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
  } catch {
    adapter = null;
  }
  if (!adapter) {
    return {
      status: "unsupported",
      memory: deviceMemoryGB === null ? "unknown" : "limited",
      webgpu: true,
      adapter: false,
      shaderF16: false,
      deviceMemoryGB,
      jsHeapLimitBytes,
      storageAvailableBytes,
      maxBufferSize: null,
      maxStorageBufferBindingSize: null,
      reasons: ["adapter_unavailable"],
    };
  }

  const reasons = [];
  const shaderF16 = adapter.features?.has("shader-f16") === true;
  const maxBufferSize = finitePositive(adapter.limits?.maxBufferSize);
  const maxStorageBufferBindingSize = finitePositive(
    adapter.limits?.maxStorageBufferBindingSize,
  );

  if (!shaderF16) reasons.push("shader_f16_unavailable");
  if (
    (maxBufferSize !== null && maxBufferSize < MIN_BUFFER_BYTES) ||
    (
      maxStorageBufferBindingSize !== null &&
      maxStorageBufferBindingSize < MIN_STORAGE_BUFFER_BYTES
    )
  ) {
    reasons.push("gpu_buffer_limits_low");
  }
  if (
    deviceMemoryGB !== null &&
    deviceMemoryGB < RECOMMENDED_DEVICE_MEMORY_GB
  ) {
    reasons.push("device_memory_low");
  }
  if (jsHeapLimitBytes !== null && jsHeapLimitBytes < MIN_STORAGE_BYTES) {
    reasons.push("js_heap_low");
  }
  if (
    storageAvailableBytes !== null &&
    storageAvailableBytes < MIN_STORAGE_BYTES
  ) {
    reasons.push("storage_low");
  }

  const hasMemorySignal =
    deviceMemoryGB !== null ||
    jsHeapLimitBytes !== null ||
    storageAvailableBytes !== null;
  const memoryLimited = reasons.some((reason) =>
    reason === "device_memory_low" ||
    reason === "js_heap_low" ||
    reason === "storage_low" ||
    reason === "gpu_buffer_limits_low"
  );
  const memory = memoryLimited
    ? "limited"
    : hasMemorySignal
      ? "enough"
      : "unknown";
  if (!hasMemorySignal) reasons.push("memory_unknown");

  const unsupported = reasons.some((reason) =>
    reason === "shader_f16_unavailable" ||
    reason === "gpu_buffer_limits_low"
  );
  return {
    status: unsupported
      ? "unsupported"
      : memory === "enough"
        ? "ready"
        : "limited",
    memory,
    webgpu: true,
    adapter: true,
    shaderF16,
    deviceMemoryGB,
    jsHeapLimitBytes,
    storageAvailableBytes,
    maxBufferSize,
    maxStorageBufferBindingSize,
    reasons,
  };
}
