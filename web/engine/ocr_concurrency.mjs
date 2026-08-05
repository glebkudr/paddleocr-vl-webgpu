const MAX_BROWSER_OCR_PARALLELISM = 4;

function finitePositive(value, fallback) {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function recommendOcrParallelism(
  capabilities = {},
  {
    hardwareConcurrency = globalThis.navigator?.hardwareConcurrency ?? 1,
    deviceMemory = globalThis.navigator?.deviceMemory ?? 0,
  } = {},
) {
  const adapter = String(capabilities.adapter_name ?? "").toLowerCase();
  const deviceType = String(capabilities.adapter_device_type ?? "").toLowerCase();
  const software = /(swiftshader|llvmpipe|software|cpu)/.test(adapter);
  if (software) {
    return Object.freeze({
      maxParallelism: 1,
      initialParallelism: 1,
      reason: "software_adapter",
    });
  }

  let maxParallelism = 1;
  let reason = "conservative_default";
  if (
    /(rtx|quadro|tesla).*(3090|4090|5090|a6000|a100|h100)/.test(adapter) ||
    /(radeon|rx).*(7900|6900)/.test(adapter) ||
    /apple.*(max|ultra)/.test(adapter)
  ) {
    maxParallelism = 4;
    reason = "large_discrete_gpu";
  } else if (
    /(rtx|quadro|tesla).*(3080|4080|5080|a5000)/.test(adapter) ||
    /(radeon|rx).*(7800|6800)/.test(adapter)
  ) {
    maxParallelism = 3;
    reason = "high_end_discrete_gpu";
  } else if (
    /(nvidia|geforce|rtx|radeon|amd rx|intel arc)/.test(adapter) ||
    deviceType === "discretegpu" ||
    deviceType === "discrete_gpu" ||
    /apple.*pro/.test(adapter)
  ) {
    maxParallelism = 2;
    reason = "parallel_capable_gpu";
  } else if (hardwareConcurrency >= 12 && deviceMemory >= 8 && deviceType !== "integratedgpu") {
    maxParallelism = 2;
    reason = "high_end_host_probe";
  }

  maxParallelism = Math.max(
    1,
    Math.min(MAX_BROWSER_OCR_PARALLELISM, maxParallelism),
  );
  return Object.freeze({
    maxParallelism,
    // The first block deliberately warms all shared resident weights alone.
    initialParallelism: 1,
    reason,
  });
}

function resultWorkUnits(outcome) {
  if (!outcome?.ok) return 0;
  const result = outcome.result ?? {};
  const generated = finitePositive(
    result.greedy?.tokenIds?.length ?? result.generatedTokens,
    1,
  );
  const vision = finitePositive(result.visionTokens, 1);
  return generated + vision * 0.2 + 8;
}

export class AdaptiveOcrConcurrency {
  #maxParallelism;
  #currentParallelism;
  #bestParallelism = 1;
  #bestRate = 0;
  #waves = [];

  constructor(profile) {
    this.profile = Object.freeze({ ...profile });
    this.#maxParallelism = Math.max(
      1,
      Math.min(
        MAX_BROWSER_OCR_PARALLELISM,
        Number(profile?.maxParallelism) || 1,
      ),
    );
    this.#currentParallelism = 1;
  }

  current(remaining = Number.POSITIVE_INFINITY) {
    return Math.max(
      1,
      Math.min(this.#currentParallelism, this.#maxParallelism, remaining),
    );
  }

  recordWave({ width, wallMs, outcomes }) {
    const safeWallMs = finitePositive(wallMs, 1);
    const failures = outcomes.filter((outcome) => !outcome?.ok).length;
    const workUnits = outcomes.reduce(
      (total, outcome) => total + resultWorkUnits(outcome),
      0,
    );
    const rate = workUnits / safeWallMs;
    const wave = Object.freeze({
      width,
      wallMs: safeWallMs,
      workUnits,
      rate,
      failures,
    });
    this.#waves.push(wave);

    if (failures > 0) {
      this.#currentParallelism = Math.max(1, Math.min(this.#bestParallelism, width - 1));
      return wave;
    }
    if (rate > this.#bestRate * 1.08 || this.#bestRate === 0) {
      this.#bestRate = rate;
      this.#bestParallelism = width;
      this.#currentParallelism = Math.min(this.#maxParallelism, width + 1);
      return wave;
    }
    if (rate < this.#bestRate * 0.9) {
      this.#currentParallelism = this.#bestParallelism;
      return wave;
    }
    this.#currentParallelism = width;
    return wave;
  }

  report() {
    return Object.freeze({
      ...this.profile,
      currentParallelism: this.#currentParallelism,
      bestParallelism: this.#bestParallelism,
      bestRate: this.#bestRate,
      waves: Object.freeze([...this.#waves]),
    });
  }
}

export class BrowserRuntimePool {
  #available;
  #waiters = [];
  #all;

  constructor(runtimes) {
    if (!Array.isArray(runtimes) || runtimes.length === 0) {
      throw new Error("browser OCR runtime pool requires at least one runtime");
    }
    this.#all = [...runtimes];
    this.#available = [...runtimes];
  }

  async acquire() {
    const runtime = this.#available.pop();
    if (runtime) return runtime;
    return new Promise((resolve) => this.#waiters.push(resolve));
  }

  release(runtime) {
    const waiter = this.#waiters.shift();
    if (waiter) {
      waiter(runtime);
      return;
    }
    this.#available.push(runtime);
  }

  async run(task) {
    const runtime = await this.acquire();
    try {
      return await task(runtime);
    } finally {
      this.release(runtime);
    }
  }

  dispose() {
    for (const runtime of this.#all) runtime.free();
    this.#all = [];
    this.#available = [];
    this.#waiters = [];
  }
}
