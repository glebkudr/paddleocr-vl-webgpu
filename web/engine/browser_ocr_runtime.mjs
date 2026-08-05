import * as runtimeWasmModule from "./pkg/pvlc_runtime_web.js";
import { runGreedyGeneration } from "./greedy_generation.mjs";
import {
  AdaptiveOcrConcurrency,
  BrowserRuntimePool,
  recommendOcrParallelism,
} from "./ocr_concurrency.mjs";
import {
  PADDLEOCR_VL_TASKS,
  recognitionOptionsForTask,
} from "./paddleocr_vl_tasks.mjs";
import {
  assembleEmbeddings,
  buildMultimodalSessionPack,
  buildVisionStackManifest,
  loadMultimodalTokenizer,
  mropePositions,
  multimodalF32Bytes,
  multimodalPromptInputIds,
  narrowMultimodalF32ToF16Bytes,
  patchEmbeddingWebGpu,
  preprocessImage,
  readMultimodalSafetensors,
  readSessionPackShardBytes,
  ropeTables,
  runSyntheticVisionStack,
  widenMultimodalF16Bytes,
} from "./multimodal_ocr.mjs";

const { default: init, WebRuntime } = runtimeWasmModule;

export const MODEL_REPOSITORY = "glebkudr/PaddleOCR-VL-1.6-WebGPU";
export const MODEL_REVISION = "5219606297d00bbd23b2ec193271af0c0b427dfc";
export const MODEL_DOWNLOAD_BYTES = 2_038_286_862;

const MODEL_BASE_URL =
  `https://huggingface.co/${MODEL_REPOSITORY}/resolve/${MODEL_REVISION}`;
const PROMPT_PREFIX_IDS = Object.freeze([
  100273, 2969, 93963, 93919, 101305,
]);
const ASSETS = Object.freeze({
  stackBase:
    `${MODEL_BASE_URL}/m7u-vision-stack-full-fp16-input-major-official`,
  visionEmbedUrl: `${MODEL_BASE_URL}/m6e10-vision-embed.safetensors`,
  projectorBase:
    `${MODEL_BASE_URL}/m7v-projector-full-fp16-input-major-official`,
  decoderPackUrl:
    `${MODEL_BASE_URL}/m7q1-decoder-stack-logits-fp16-official.pvlc`,
  embedTokensUrl: `${MODEL_BASE_URL}/m6e9-embed-tokens.f32`,
  tokenizerUrl: `${MODEL_BASE_URL}/m6e9-tokenizer.json`,
});

const completedAssetUrls = new Set();
const assetDownloadBytes = new Map();
const MODEL_CACHE_NAME = `sotaocr-free-ocr-${MODEL_REVISION}`;

function emitProgress(onProgress, update) {
  if (typeof onProgress === "function") {
    onProgress(update);
  }
}

function modelLoadedBytes() {
  let total = 0;
  for (const value of assetDownloadBytes.values()) {
    total += value;
  }
  return Math.min(total, MODEL_DOWNLOAD_BYTES);
}

async function fetchBytes(url, label, onProgress) {
  let modelCache = null;
  let response = null;
  let fromPersistentCache = false;
  if ("caches" in globalThis) {
    try {
      modelCache = await caches.open(MODEL_CACHE_NAME);
      response = await modelCache.match(url);
      fromPersistentCache = response !== undefined;
    } catch {
      modelCache = null;
      response = null;
    }
  }
  if (!response) {
    response = await fetch(url, { cache: "force-cache" });
  }
  if (!response.ok) {
    throw new Error(`${label} fetch failed: HTTP ${response.status}`);
  }
  const cacheWrite = modelCache && !fromPersistentCache
    ? modelCache.put(url, response.clone()).catch(() => {})
    : Promise.resolve();

  const totalBytes = Number(response.headers.get("content-length")) || null;
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    await cacheWrite;
    if (!completedAssetUrls.has(url)) {
      assetDownloadBytes.set(url, bytes.byteLength);
      completedAssetUrls.add(url);
    }
    emitProgress(onProgress, {
      kind: "download",
      asset: label,
      assetLoadedBytes: bytes.byteLength,
      assetTotalBytes: totalBytes,
      modelLoadedBytes: modelLoadedBytes(),
      modelTotalBytes: MODEL_DOWNLOAD_BYTES,
      fromPersistentCache,
    });
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loadedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loadedBytes += value.byteLength;
    if (!completedAssetUrls.has(url)) {
      assetDownloadBytes.set(url, loadedBytes);
    }
    emitProgress(onProgress, {
      kind: "download",
      asset: label,
      assetLoadedBytes: loadedBytes,
      assetTotalBytes: totalBytes,
      modelLoadedBytes: modelLoadedBytes(),
      modelTotalBytes: MODEL_DOWNLOAD_BYTES,
      fromPersistentCache,
    });
  }

  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  if (!completedAssetUrls.has(url)) {
    assetDownloadBytes.set(url, loadedBytes);
    completedAssetUrls.add(url);
  }
  await cacheWrite;
  return bytes;
}

class BrowserModelAssets {
  #memoized = new Map();

  #once(key, loader) {
    if (!this.#memoized.has(key)) {
      const pending = Promise.resolve()
        .then(loader)
        .catch((error) => {
          this.#memoized.delete(key);
          throw error;
        });
      this.#memoized.set(key, pending);
    }
    return this.#memoized.get(key);
  }

  bytes(url, label, onProgress) {
    return this.#once(
      `bytes:${url}`,
      () => fetchBytes(url, label, onProgress),
    );
  }

  visionEmbed(onProgress) {
    return this.#once("parsed:vision-embed", async () =>
      readMultimodalSafetensors(
        await this.bytes(
          ASSETS.visionEmbedUrl,
          "Vision embeddings",
          onProgress,
        ),
        "vision embed",
      )
    );
  }

  visionManifest(onProgress) {
    return this.#once("parsed:vision-manifest", async () =>
      JSON.parse(new TextDecoder().decode(
        await this.bytes(
          `${ASSETS.stackBase}/manifest.json`,
          "Vision manifest",
          onProgress,
        ),
      ))
    );
  }

  projectorDescriptor(onProgress) {
    return this.#once("parsed:projector-descriptor", async () => {
      const json = new TextDecoder().decode(
        await this.bytes(
          `${ASSETS.projectorBase}/manifest.json`,
          "Projector manifest",
          onProgress,
        ),
      );
      return Object.freeze({ json, value: JSON.parse(json) });
    });
  }

  tokenEmbeddings(onProgress) {
    return this.#once("parsed:token-embeddings", async () =>
      new Float32Array(
        (await this.bytes(
          ASSETS.embedTokensUrl,
          "Token embeddings",
          onProgress,
        )).buffer,
      )
    );
  }

  decoderShards(runtime, onProgress) {
    return this.#once("parsed:decoder-shards", async () =>
      readSessionPackShardBytes(
        await this.bytes(
          ASSETS.decoderPackUrl,
          "Decoder weights",
          onProgress,
        ),
        (bytes) => runtime.blake3_bytes_hex(bytes),
      )
    );
  }

  tokenizer(onProgress) {
    return this.#once("parsed:tokenizer", async () =>
      loadMultimodalTokenizer(
        await this.bytes(
          ASSETS.tokenizerUrl,
          "Tokenizer",
          onProgress,
        ),
      )
    );
  }

  preload(onProgress) {
    return Object.freeze({
      visionEmbed: this.visionEmbed(onProgress),
      visionManifest: this.visionManifest(onProgress),
      projectorDescriptor: this.projectorDescriptor(onProgress),
      tokenEmbeddings: this.tokenEmbeddings(onProgress),
      tokenizer: this.tokenizer(onProgress),
    });
  }

  releaseDecoderShards() {
    this.#memoized.delete("parsed:decoder-shards");
    this.#memoized.delete(`bytes:${ASSETS.decoderPackUrl}`);
  }

  release() {
    this.#memoized.clear();
  }
}

function canvasRgb(canvas) {
  const imageData = canvas
    .getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, canvas.width, canvas.height);
  const rgb = new Uint8Array(canvas.width * canvas.height * 3);
  for (let index = 0; index < canvas.width * canvas.height; index += 1) {
    rgb[index * 3] = imageData.data[index * 4];
    rgb[index * 3 + 1] = imageData.data[index * 4 + 1];
    rgb[index * 3 + 2] = imageData.data[index * 4 + 2];
  }
  return rgb;
}

async function runCanvasInference(
  runtime,
  assets,
  canvas,
  options,
  onProgress,
) {
  const startedAt = performance.now();
  const residentState = typeof runtime.decoder_stack_resident_weights_json === "function"
    ? JSON.parse(runtime.decoder_stack_resident_weights_json())
    : { ready: false };
  const canUseResidentBegin =
    residentState.ready === true &&
    typeof runtime.begin_decoder_stack_session_resident === "function";
  let decoderShardsPromise = canUseResidentBegin
    ? null
    : assets.decoderShards(runtime, onProgress);
  const preloaded = assets.preload(onProgress);
  const mark = (stage, detail) => {
    emitProgress(onProgress, { kind: "inference", stage, detail });
  };

  const rgb = canvasRgb(canvas);
  const { pixelValues, gridThw } = preprocessImage(
    rgb,
    canvas.height,
    canvas.width,
  );
  const tokens = gridThw[0] * gridThw[1] * gridThw[2];
  mark("preprocess", `${tokens} vision tokens`);

  const visionEmbed = await preloaded.visionEmbed;
  const embedded = await patchEmbeddingWebGpu(runtime, pixelValues, {
    weight: visionEmbed("patch_embedding_weight"),
    bias: visionEmbed("patch_embedding_bias"),
    positionEmbedding: visionEmbed("position_embedding_weight"),
    gridThw,
  });
  mark("patch_embedding", "Patch and position embeddings ready");

  const officialManifest = await preloaded.visionManifest;
  const layerSuffix = officialManifest.matrix_weight_storage === "f16"
    ? ".bin"
    : ".f32";
  const loadShard = (id) => fetchBytes(
    `${ASSETS.stackBase}/${id}${layerSuffix}`,
    `Vision ${id}`,
    onProgress,
  );
  const activationStorage = officialManifest.activation_storage ?? "f32";
  const visionInputBytes = activationStorage === "f16"
    ? narrowMultimodalF32ToF16Bytes(
      embedded.output,
      "vision stack input embeddings",
    )
    : multimodalF32Bytes(embedded.output);
  const visionManifest = buildVisionStackManifest({
    tokens,
    weightShards: officialManifest.shards.slice(1),
    inputBlake3: runtime.blake3_bytes_hex(visionInputBytes),
    caseId: "sotaocr.free_ocr_browser",
    matrixWeightStorage:
      officialManifest.matrix_weight_storage ?? "f32",
    matrixWeightLayout:
      officialManifest.matrix_weight_layout ?? "output_major",
    vectorWeightStorage:
      officialManifest.vector_weight_storage ?? "f32",
    activationStorage,
    checkpointLayers: [],
  });

  const {
    json: projectorDescriptorJson,
    value: projectorDescriptor,
  } = await preloaded.projectorDescriptor;
  if (!runtime.has_projector_f16_resident_weights(projectorDescriptorJson)) {
    runtime.prepare_projector_f16_resident_weights(
      projectorDescriptorJson,
      await fetchBytes(
        `${ASSETS.projectorBase}/weights.projector.bin`,
        "Projector weights",
        onProgress,
      ),
    );
  }

  const imageTokens = tokens / 4;
  const projectorF16 = Object.freeze({
    descriptorJson: projectorDescriptorJson,
    imageGridThwJson: JSON.stringify([gridThw]),
    outputTokens: imageTokens,
    outputSize: projectorDescriptor.output_size,
  });
  const visionExecution = await runSyntheticVisionStack(
    runtime,
    visionManifest,
    loadShard,
    visionInputBytes,
    { gridThw, projectorF16 },
  );
  mark("vision_encoder", "Vision encoder complete");

  let projectorBytes = visionExecution.projectorBytes;
  if (!visionExecution.timings.projectorChained) {
    const visionFinal = activationStorage === "f16"
      ? visionExecution.bytes
      : narrowMultimodalF32ToF16Bytes(
        new Float32Array(
          visionExecution.bytes.buffer,
          visionExecution.bytes.byteOffset,
          visionExecution.bytes.byteLength / 4,
        ),
        "vision stack final activation",
      );
    const projected = await runtime.run_projector_f16_resident_bytes(
      projectorDescriptorJson,
      projectorF16.imageGridThwJson,
      visionFinal,
    );
    projectorBytes = projected.checkpoint_bytes;
  }
  if (
    !(projectorBytes instanceof Uint8Array) ||
    projectorBytes.byteLength !==
      imageTokens * projectorDescriptor.output_size * 2
  ) {
    throw new Error("FP16 projector readback drifted");
  }
  const projectorFinal = new Float32Array(
    widenMultimodalF16Bytes(
      projectorBytes,
      "FP16 projector output",
    ).buffer,
  );
  mark("projector", "Visual features projected");

  const inputIds = multimodalPromptInputIds(gridThw, {
    prefixIds: PROMPT_PREFIX_IDS,
    suffixIds: options.promptSuffixIds,
  });
  const promptTokens = inputIds.length;
  const capacity = promptTokens + options.maxGeneratedTokens;
  const embedTokens = await preloaded.tokenEmbeddings;
  const tokenEmbedding = Object.freeze({
    values: embedTokens,
    gather(id) {
      return embedTokens.subarray(id * 1024, (id + 1) * 1024);
    },
  });
  const assembled = assembleEmbeddings(
    tokenEmbedding,
    projectorFinal,
    inputIds,
  );
  const { positionIds, ropeDelta } = mropePositions(inputIds, gridThw);
  const rope = ropeTables(positionIds, ropeDelta, { capacity });
  mark("prompt", `${promptTokens} prompt tokens`);

  let sessionPack = null;
  if (!canUseResidentBegin) {
    const decoderShards = await decoderShardsPromise;
    const sourceDescriptor = JSON.parse(new TextDecoder().decode(
      decoderShards.get("ir.decoder_stack_00"),
    ));
    const shardView = (id) => decoderShards.get(id);
    sessionPack = buildMultimodalSessionPack({
      descriptor: { prefix_tokens: 0, cache_capacity: capacity },
      weights: {
        norm1: shardView("weights.input_layernorm"),
        q: shardView("weights.q_proj"),
        k: shardView("weights.k_proj"),
        v: shardView("weights.v_proj"),
        o: shardView("weights.o_proj"),
        ropeCos: rope.cos,
        ropeSin: rope.sin,
        norm2: shardView("weights.post_attention_layernorm"),
        gate: shardView("weights.gate_proj"),
        up: shardView("weights.up_proj"),
        down: shardView("weights.down_proj"),
        finalNorm: shardView("weights.final_layernorm"),
        lmHead: shardView("weights.lm_head"),
      },
      blake3Hex: (bytes) => runtime.blake3_bytes_hex(bytes),
      caseId: "sotaocr.free_ocr_browser.decoder",
      weightStorage: "f16",
      checkpointIdentity: {
        blake3: sourceDescriptor.checkpoint_blake3,
        bytes: sourceDescriptor.checkpoint_bytes,
      },
    });
  }

  const descriptor = {
    schema_version: 1,
    layers: 18,
    hidden_size: 1024,
    intermediate_size: 3072,
    query_heads: 16,
    key_value_heads: 2,
    head_dim: 128,
    query_width: 2048,
    key_value_width: 256,
    prefix_tokens: 0,
    cache_capacity: capacity,
    mrope_sections: [16, 24, 24],
    rms_norm_epsilon: 1e-5,
    prefill_tokens: promptTokens,
    vocab_size: 103424,
  };

  let decoderSessionStarted = false;
  const decoderSetupStartedAt = performance.now();
  let decoderCreationDiagnostics = null;
  try {
    const descriptorJson = JSON.stringify(descriptor);
    decoderCreationDiagnostics = JSON.parse(canUseResidentBegin
      ? await runtime.begin_decoder_stack_session_resident(
        descriptorJson,
        multimodalF32Bytes(rope.cos),
        multimodalF32Bytes(rope.sin),
      )
      : await runtime.begin_decoder_stack_session(
        descriptorJson,
        sessionPack,
        new Uint8Array(18 * capacity * 256 * 4),
        new Uint8Array(18 * capacity * 256 * 4),
      ));
    decoderSessionStarted = true;
    sessionPack = null;
    decoderShardsPromise = null;
    if (!canUseResidentBegin) {
      assets.releaseDecoderShards();
    }
    const decoderSetupMs = performance.now() - decoderSetupStartedAt;
    await runtime.prefill_decoder_stack_session(
      multimodalF32Bytes(assembled),
    );
    mark("decoder_prefill", "Decoder prefill complete");

    const generationStartedAt = performance.now();
    const greedy = await runGreedyGeneration(runtime, {
      embedding: embedTokens,
      eosTokenId: 2,
      maxSteps: options.maxGeneratedTokens,
      cacheCapacity: capacity,
      initialCacheTokens: promptTokens,
    });
    const generationMs = performance.now() - generationStartedAt;
    const tokensPerSecond = generationMs === 0
      ? 0
      : greedy.tokenIds.length * 1000 / generationMs;
    mark("generation", `${greedy.tokenIds.length} generated tokens`);

    const tokenizer = await preloaded.tokenizer;
    return {
      greedy,
      text: tokenizer.detokenize([...greedy.tokenIds]),
      generationMs,
      tokensPerSecond,
      visionTokens: tokens,
      promptTokens,
      gpuTop1: greedy.gpuTop1,
      logitsReadbackBytes: greedy.logitsReadbackBytes,
      selectionQueueWallTimeNs: greedy.selectionQueueWallTimeNs,
      decoderSetupMs,
      decoderResidentWeightBytes:
        JSON.parse(runtime.decoder_stack_resident_weights_json())
          .resident_weight_bytes,
      decoderResidentWeightCacheHit:
        canUseResidentBegin,
      elapsedMs: performance.now() - startedAt,
    };
  } finally {
    if (decoderSessionStarted) {
      runtime.abort_decoder_stack_session();
    }
  }
}

export {
  PADDLEOCR_VL_TASKS,
  recognitionOptionsForTask,
};

export class PaddleOcrVlEngine {
  #runtimePool;
  #concurrency;
  #assets;
  #disposed = false;

  constructor(runtimes, concurrency) {
    this.#runtimePool = new BrowserRuntimePool(runtimes);
    this.#concurrency = concurrency;
    this.#assets = new BrowserModelAssets();
  }

  recognitionConcurrency(remaining = Number.POSITIVE_INFINITY) {
    if (this.#disposed) {
      throw new Error("PaddleOCR-VL engine has been disposed");
    }
    return this.#concurrency.current(remaining);
  }

  recordRecognitionWave(wave) {
    if (this.#disposed) {
      throw new Error("PaddleOCR-VL engine has been disposed");
    }
    return this.#concurrency.recordWave(wave);
  }

  async recognizeCanvas(
    canvas,
    {
      task = "ocr",
      promptSuffixIds,
      maxGeneratedTokens = 512,
      onProgress = () => {},
    } = {},
  ) {
    if (this.#disposed) {
      throw new Error("PaddleOCR-VL engine has been disposed");
    }
    if (
      !canvas ||
      !Number.isInteger(canvas.width) ||
      !Number.isInteger(canvas.height) ||
      canvas.width <= 0 ||
      canvas.height <= 0 ||
      typeof canvas.getContext !== "function"
    ) {
      throw new TypeError("recognizeCanvas() expects a non-empty canvas");
    }
    const taskOptions = recognitionOptionsForTask(task, {
      maxGeneratedTokens,
    });
    return this.#runtimePool.run((runtime) =>
      runCanvasInference(
        runtime,
        this.#assets,
        canvas,
        {
          ...taskOptions,
          promptSuffixIds:
            promptSuffixIds ?? taskOptions.promptSuffixIds,
        },
        onProgress,
      )
    );
  }

  async recognizeImage(image, options = {}) {
    if (!(image instanceof Blob)) {
      throw new TypeError("recognizeImage() expects a browser File or Blob");
    }
    const bitmap = await createImageBitmap(image);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Could not create an image canvas");
      context.drawImage(bitmap, 0, 0);
      return await this.recognizeCanvas(canvas, options);
    } finally {
      bitmap.close();
    }
  }

  async dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#assets.release();
    this.#runtimePool.dispose();
  }
}

export async function createPaddleOcrVlEngine() {
  await init();
  const primaryRuntime = await WebRuntime.create();
  const capabilities = JSON.parse(primaryRuntime.capabilities_json());
  const profile = recommendOcrParallelism(capabilities);
  const extraRuntimes = await Promise.all(
    Array.from(
      { length: profile.maxParallelism - 1 },
      () => WebRuntime.create(),
    ),
  );
  return new PaddleOcrVlEngine(
    [primaryRuntime, ...extraRuntimes],
    new AdaptiveOcrConcurrency(profile),
  );
}
