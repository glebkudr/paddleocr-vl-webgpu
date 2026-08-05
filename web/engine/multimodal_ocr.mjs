// PVLC M6e10 caller-owned multimodal assembly pipeline
// (docs/m6e10_browser_multimodal_assembly_contract.md).
//
// The host-side image-to-prefill assembly: preprocessing (smart_resize, the
// Pillow bicubic mirror, rescale/normalize, patch extraction), patch
// embedding (the accepted patch_projection_f32 semantics plus bilinear
// position interpolation), prompt ids (per-case pinned text sides plus the
// image-run expansion), embedding assembly, M-RoPE positions, and the
// standard RoPE tables (rope_theta 500000, sections [16, 24, 24], duplicated
// halves, decode-continuation rows from the pinned rope delta). The module
// is self-contained within the runner layer: it imports NOTHING from
// web/tests — the test-side oracle (web/tests/support/m6e10_multimodal_oracle.mjs)
// remains the comparison authority; this module is the production mirror
// composed by the demo page and exercised by the M6e10 browser gate.

export const MULTIMODAL = Object.freeze({
  factor: 28,
  minPixels: 112896,
  maxPixels: 1003520,
  patchSize: 14,
  mergeSize: 2,
  imageMean: 0.5,
  imageStd: 0.5,
  rescale: 1 / 255,
  ropeTheta: 500000,
  headDim: 128,
  mropeSections: Object.freeze([16, 24, 24]),
  imageTokenId: 100295,
  visionStartTokenId: 101305,
  visionHidden: 1152,
  visionHeadDim: 72,
  visionRopeTheta: 10000,
  decoderHidden: 1024,
  positionGrid: 27,
});

function fail(message) {
  throw new Error(`multimodal assembly: ${message}`);
}

export function enqueueVisionStackLayer(runtime, shardId, payload) {
  if (
    typeof runtime?.enqueue_vision_encoder_stack_sharded_layer_json ===
    "function"
  ) {
    return {
      streaming: true,
      status: runtime.enqueue_vision_encoder_stack_sharded_layer_json(
        shardId,
        payload,
      ),
    };
  }
  if (
    typeof runtime?.run_vision_encoder_stack_sharded_layer_json === "function"
  ) {
    return {
      streaming: false,
      completion: runtime.run_vision_encoder_stack_sharded_layer_json(
        shardId,
        payload,
      ),
    };
  }
  throw new Error("vision stack layer API is unavailable");
}

export async function preflightVisionStackShards(
  runtime,
  shards,
  loadShard,
  inputBytes,
) {
  if (
    typeof runtime?.preflight_vision_encoder_stack_manifest_shard_json ===
    "function"
  ) {
    for (const shard of shards) {
      runtime.preflight_vision_encoder_stack_manifest_shard_json(shard.id);
    }
    return "manifest";
  }
  if (
    typeof runtime?.preflight_vision_encoder_stack_shard_json === "function"
  ) {
    for (const [index, shard] of shards.entries()) {
      const payload = index === 0 ? inputBytes : await loadShard(shard.id);
      runtime.preflight_vision_encoder_stack_shard_json(shard.id, payload);
    }
    return "payload";
  }
  throw new Error("vision stack preflight API is unavailable");
}

// One synthetic-oracle vision-stack run over caller-computed embeddings.
// Manifest preflight proves only ordered intent; each payload remains fully
// authenticated by the runtime immediately before its execution GPU effects.
export function visionRope2dTables(gridThw, {
  headDim = MULTIMODAL.visionHeadDim,
  theta = MULTIMODAL.visionRopeTheta,
} = {}) {
  const grids = Array.isArray(gridThw?.[0]) ? gridThw : [gridThw];
  if (
    grids.length === 0 ||
    !Number.isSafeInteger(headDim) ||
    headDim <= 0 ||
    headDim % 4 !== 0 ||
    !Number.isFinite(theta) ||
    theta <= 0 ||
    grids.some((grid) =>
      !Array.isArray(grid) ||
      grid.length !== 3 ||
      grid.some((value) => !Number.isSafeInteger(value) || value <= 0)
    )
  ) {
    fail("vision RoPE geometry is invalid");
  }
  let tokens = 0;
  for (const [temporal, height, width] of grids) {
    const gridTokens = temporal * height * width;
    if (!Number.isSafeInteger(gridTokens) ||
        !Number.isSafeInteger(tokens + gridTokens)) {
      fail("vision RoPE token geometry overflowed");
    }
    tokens += gridTokens;
  }
  const pairCount = headDim / 2;
  const frequencyCount = pairCount / 2;
  const tableElements = tokens * pairCount;
  if (!Number.isSafeInteger(tableElements)) {
    fail("vision RoPE table geometry overflowed");
  }
  const cos = new Float32Array(tableElements);
  const sin = new Float32Array(tableElements);
  let token = 0;
  for (const [temporal, height, width] of grids) {
    for (let time = 0; time < temporal; time += 1) {
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const tableStart = token * pairCount;
          for (let pair = 0; pair < pairCount; pair += 1) {
            const position = pair < frequencyCount ? y : x;
            const frequency = pair % frequencyCount;
            const inverseFrequency = theta ** (-(2 * frequency) / pairCount);
            const angle = position * inverseFrequency;
            cos[tableStart + pair] = Math.cos(angle);
            sin[tableStart + pair] = Math.sin(angle);
          }
          token += 1;
        }
      }
    }
  }
  return { cos, sin };
}

export async function runSyntheticVisionStack(
  runtime,
  manifest,
  loadShard,
  inputBytes,
  { gridThw, projectorF16 } = {},
) {
  const startedAt = performance.now();
  const manifestJson = visionStackManifestJson(manifest);
  const qkvPolicy = manifest.matrix_weight_storage === "f16"
    ? "disabled"
    : "required";
  const qkvSelection = runtime.compile_vision_encoder_stack_qkv_selection(
    manifestJson,
    qkvPolicy,
  );
  const residentWeightApiAvailable =
    manifest.matrix_weight_storage === "f16" &&
    manifest.matrix_weight_layout === "input_major" &&
    typeof runtime?.has_vision_encoder_stack_resident_weights === "function" &&
    typeof runtime
      ?.begin_vision_encoder_stack_sharded_resident_with_activation_strategy_and_qkv_selection_json ===
      "function" &&
    typeof runtime
      ?.enqueue_vision_encoder_stack_sharded_resident_layer_json ===
      "function" &&
    typeof runtime?.finish_vision_encoder_stack_sharded_resident === "function";
  const residentCacheHit = residentWeightApiAvailable
    ? runtime.has_vision_encoder_stack_resident_weights(manifestJson)
    : false;
  const beginJson = residentWeightApiAvailable
    ? runtime
      .begin_vision_encoder_stack_sharded_resident_with_activation_strategy_and_qkv_selection_json(
        manifestJson,
        "static_arena_alias",
        qkvSelection,
      )
    : runtime
      .begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json(
        manifestJson,
        "static_arena_alias",
        qkvSelection,
      );
  const begin = JSON.parse(
    beginJson,
  );
  if (begin.phase !== "preflight") {
    throw new Error("vision stack begin drifted");
  }
  if (gridThw !== undefined) {
    if (
      typeof runtime?.configure_vision_encoder_stack_spatial_rope_f32 !==
      "function"
    ) {
      throw new Error("vision stack spatial RoPE API is unavailable");
    }
    const { cos, sin } = visionRope2dTables(gridThw);
    runtime.configure_vision_encoder_stack_spatial_rope_f32(cos, sin);
  }
  const preflightStartedAt = performance.now();
  await preflightVisionStackShards(runtime, manifest.shards, loadShard, inputBytes);
  const preflightMs = performance.now() - preflightStartedAt;
  const executionStartedAt = performance.now();
  await runtime.start_vision_encoder_stack_sharded_json(
    manifest.shards[0].id,
    inputBytes,
  );
  const layerMs = [];
  const layerLoadMs = [];
  const layerRuntimeMs = [];
  for (let layer = 0; layer < manifest.layer_count; layer += 1) {
    const shard = manifest.shards[layer + 1];
    const layerStartedAt = performance.now();
    const payload = residentCacheHit ? undefined : await loadShard(shard.id);
    const runtimeStartedAt = performance.now();
    const submitted = residentCacheHit
      ? {
        streaming: true,
        status:
          runtime.enqueue_vision_encoder_stack_sharded_resident_layer_json(
            shard.id,
          ),
      }
      : enqueueVisionStackLayer(runtime, shard.id, payload);
    if (!submitted.streaming) await submitted.completion;
    layerLoadMs.push(runtimeStartedAt - layerStartedAt);
    layerRuntimeMs.push(performance.now() - runtimeStartedAt);
    layerMs.push(performance.now() - layerStartedAt);
  }
  const finishStartedAt = performance.now();
  const lastShard = manifest.shards.at(-1);
  const projectorChained =
    residentCacheHit &&
    projectorF16 !== undefined &&
    typeof runtime
      ?.finish_vision_encoder_stack_sharded_resident_with_projector_f16 ===
      "function";
  const finished = projectorChained
    ? await runtime
      .finish_vision_encoder_stack_sharded_resident_with_projector_f16(
        lastShard.id,
        projectorF16.descriptorJson,
        projectorF16.imageGridThwJson,
      )
    : residentCacheHit
      ? await runtime.finish_vision_encoder_stack_sharded_resident(lastShard.id)
    : await runtime.finish_vision_encoder_stack_sharded(
      lastShard.id,
      await loadShard(lastShard.id),
    );
  const finishMs = performance.now() - finishStartedAt;
  const activationBytes =
    (manifest.activation_storage ?? "f32") === "f16" ? 2 : 4;
  const planeBytes =
    manifest.tokens * MULTIMODAL.visionHidden * activationBytes;
  const expectedReadbackBytes = projectorChained
    ? projectorF16.outputTokens * projectorF16.outputSize * 2
    : (manifest.checkpoint_layers.length + 1) * planeBytes;
  if (
    !(finished.checkpoint_bytes instanceof Uint8Array) ||
    finished.checkpoint_bytes.byteLength !== expectedReadbackBytes
  ) {
    throw new Error("vision stack finish readback drifted");
  }
  return {
    bytes: projectorChained
      ? null
      : finished.checkpoint_bytes.subarray(
        finished.checkpoint_bytes.byteLength - planeBytes,
      ),
    projectorBytes: projectorChained ? finished.checkpoint_bytes : null,
    timings: {
      totalMs: performance.now() - startedAt,
      preflightMs,
      executionMs: performance.now() - executionStartedAt,
      finishMs,
      layerMs,
      layerLoadMs,
      layerRuntimeMs,
      residentCacheHit,
      projectorChained,
      finishDiagnostics: JSON.parse(finished.diagnostics_json),
    },
  };
}

export function widenMultimodalF16Bytes(bytes, label = "F16 tensor") {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength % 2 !== 0) {
    fail(`${label} byte length is not F16-aligned`);
  }
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(bytes.byteLength / 2);
  const bitScratch = new Uint32Array(1);
  const floatScratch = new Float32Array(bitScratch.buffer);
  for (let index = 0; index < values.length; index += 1) {
    const bits = source.getUint16(index * 2, true);
    const sign = (bits & 0x8000) << 16;
    const exponent = (bits >>> 10) & 0x1f;
    const fraction = bits & 0x03ff;
    if (exponent === 0) {
      if (fraction === 0) {
        bitScratch[0] = sign;
        values[index] = floatScratch[0];
      } else {
        values[index] = (sign === 0 ? 1 : -1) * 2 ** -14 * (fraction / 1024);
      }
    } else if (exponent === 0x1f) {
      bitScratch[0] = sign | 0x7f800000 | (fraction << 13);
      values[index] = floatScratch[0];
    } else {
      bitScratch[0] = sign | ((exponent + 112) << 23) | (fraction << 13);
      values[index] = floatScratch[0];
    }
    if (!Number.isFinite(values[index])) {
      fail(`${label} contains a nonfinite F16 element at ${index}`);
    }
  }
  return new Uint8Array(values.buffer);
}

export function narrowMultimodalF32ToF16Bytes(
  values,
  label = "F32 tensor",
) {
  if (!(values instanceof Float32Array)) {
    fail(`${label} must be a Float32Array`);
  }
  const output = new Uint8Array(values.length * 2);
  const destination = new DataView(output.buffer);
  const bitScratch = new Uint32Array(1);
  const floatScratch = new Float32Array(bitScratch.buffer);
  const roundShiftToEven = (value, shift) => {
    const divisor = 2 ** shift;
    const quotient = Math.floor(value / divisor);
    const remainder = value - quotient * divisor;
    const halfway = divisor / 2;
    return quotient +
      Number(remainder > halfway || (remainder === halfway && quotient % 2 === 1));
  };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      fail(`${label} contains a nonfinite F32 element at ${index}`);
    }
    floatScratch[0] = value;
    const bits = bitScratch[0];
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    const fraction = bits & 0x7fffff;
    let half = sign;
    if (exponent === 0) {
      half = sign;
    } else if (exponent < 103) {
      half = sign;
    } else if (exponent < 113) {
      half = sign | roundShiftToEven(
        fraction | 0x800000,
        126 - exponent,
      );
    } else {
      let halfExponent = exponent - 112;
      let halfFraction = roundShiftToEven(fraction, 13);
      if (halfFraction === 0x400) {
        halfExponent += 1;
        halfFraction = 0;
      }
      if (halfExponent >= 31) {
        fail(`${label} overflows F16 at element ${index}`);
      }
      half = sign | (halfExponent << 10) | halfFraction;
    }
    destination.setUint16(index * 2, half, true);
  }
  return output;
}

export function readMultimodalSafetensors(bytes, label = "safetensors") {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < 8) {
    fail(`${label} must be a nonempty Uint8Array`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const headerEnd = 8 + headerLength;
  if (!Number.isSafeInteger(headerLength) || headerLength <= 0 || headerEnd > bytes.byteLength) {
    fail(`${label} header length drifted`);
  }
  const header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
    bytes.subarray(8, headerEnd),
  ));
  const bitScratch = new Uint32Array(1);
  const floatScratch = new Float32Array(bitScratch.buffer);
  return (name) => {
    const info = header[name];
    if (info === undefined) fail(`${label} tensor ${name} is missing`);
    if (!Array.isArray(info.shape) ||
        !info.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0) ||
        !Array.isArray(info.data_offsets) ||
        info.data_offsets.length !== 2) {
      fail(`${label} tensor ${name} metadata drifted`);
    }
    const elements = info.shape.reduce((total, dimension) => total * dimension, 1);
    const [begin, end] = info.data_offsets;
    const width = info.dtype === "F32" ? 4 : info.dtype === "BF16" || info.dtype === "F16" ? 2 : 0;
    if (width === 0 || begin < 0 || end !== begin + elements * width || headerEnd + end > bytes.byteLength) {
      fail(`${label} tensor ${name} payload drifted`);
    }
    if (info.dtype === "F32") {
      return new Float32Array(bytes.buffer, bytes.byteOffset + headerEnd + begin, elements);
    }
    if (info.dtype === "F16") {
      const widened = widenMultimodalF16Bytes(
        bytes.subarray(headerEnd + begin, headerEnd + end),
        `${label} tensor ${name}`,
      );
      return new Float32Array(widened.buffer);
    }
    const values = new Float32Array(elements);
    for (let index = 0; index < elements; index += 1) {
      const bits = view.getUint16(headerEnd + begin + index * 2, true);
      bitScratch[0] = bits << 16;
      values[index] = floatScratch[0];
    }
    return values;
  };
}

// ---------------------------------------------------------------------------
// smart_resize (image_processing_paddleocr_vl.py): factor-divisible
// dimensions, pixel budget [min_pixels, max_pixels], aspect preserved as
// closely as possible, Python round-half-even semantics.
function pythonRound(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function smartResize(height, width, {
  factor = MULTIMODAL.factor,
  minPixels = MULTIMODAL.minPixels,
  maxPixels = MULTIMODAL.maxPixels,
  patchSize = MULTIMODAL.patchSize,
} = {}) {
  if (!Number.isSafeInteger(height) || !Number.isSafeInteger(width) ||
      height <= 0 || width <= 0) {
    fail("smart_resize dimensions must be positive integers");
  }
  let resizedHeight = height;
  let resizedWidth = width;
  if (resizedHeight < factor) {
    resizedWidth = pythonRound((resizedWidth * factor) / resizedHeight);
    resizedHeight = factor;
  }
  if (resizedWidth < factor) {
    resizedHeight = pythonRound((resizedHeight * factor) / resizedWidth);
    resizedWidth = factor;
  }
  if (Math.max(resizedHeight, resizedWidth) / Math.min(resizedHeight, resizedWidth) > 200) {
    fail("smart_resize absolute aspect ratio must be smaller than 200");
  }
  let heightBar = pythonRound(resizedHeight / factor) * factor;
  let widthBar = pythonRound(resizedWidth / factor) * factor;
  if (heightBar * widthBar > maxPixels) {
    const beta = Math.sqrt((resizedHeight * resizedWidth) / maxPixels);
    heightBar = Math.floor(resizedHeight / beta / factor) * factor;
    widthBar = Math.floor(resizedWidth / beta / factor) * factor;
  } else if (heightBar * widthBar < minPixels) {
    const beta = Math.sqrt(minPixels / (resizedHeight * resizedWidth));
    heightBar = Math.ceil((resizedHeight * beta) / factor) * factor;
    widthBar = Math.ceil((resizedWidth * beta) / factor) * factor;
  }
  return {
    height: heightBar,
    width: widthBar,
    gridThw: [1, heightBar / patchSize, widthBar / patchSize],
  };
}

// ---------------------------------------------------------------------------
// The Pillow bicubic mirror (the HF image processor resizes with PIL
// Image.resize(BICUBIC)): Keys kernel a = -0.5, support 2 (widened by the
// scale on downscale), window built from the center, weights normalized, and
// the two-pass pipeline — HORIZONTAL (width) first, then VERTICAL (height) —
// with the 8-bit quantization of the intermediate and the final image.
function bicubicKernel(value) {
  const a = -0.5;
  const x = Math.abs(value);
  if (x < 1) {
    return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1;
  }
  if (x < 2) {
    return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a;
  }
  return 0;
}

export function bicubicAxisPlan(sourceSize, targetSize) {
  if (!Number.isSafeInteger(sourceSize) || sourceSize <= 0 ||
      !Number.isSafeInteger(targetSize) || targetSize <= 0) {
    fail("bicubic axis sizes must be positive integers");
  }
  const scale = sourceSize / targetSize;
  const filterScale = Math.max(scale, 1);
  const support = 2 * filterScale;
  const plan = [];
  for (let index = 0; index < targetSize; index += 1) {
    const center = (index + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), sourceSize);
    const weights = [];
    let total = 0;
    for (let source = xmin; source < xmax; source += 1) {
      const weight = bicubicKernel((source - center + 0.5) / filterScale);
      weights.push(weight);
      total += weight;
    }
    if (weights.length === 0 || total === 0) {
      fail("bicubic weights collapsed to zero");
    }
    for (let index2 = 0; index2 < weights.length; index2 += 1) {
      weights[index2] /= total;
    }
    plan.push(Object.freeze({ xmin, weights: Object.freeze(weights) }));
  }
  return Object.freeze(plan);
}

export function bicubicResizeF32(input, sourceHeight, sourceWidth, targetHeight, targetWidth, channels = 3) {
  if (input.length !== sourceHeight * sourceWidth * channels) {
    fail("bicubic input length drifted");
  }
  const widthPlan = bicubicAxisPlan(sourceWidth, targetWidth);
  const heightPlan = bicubicAxisPlan(sourceHeight, targetHeight);
  const intermediate = new Float64Array(sourceHeight * targetWidth * channels);
  for (let y = 0; y < sourceHeight; y += 1) {
    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const { xmin, weights } = widthPlan[targetX];
      for (let channel = 0; channel < channels; channel += 1) {
        let sum = 0;
        for (let k = 0; k < weights.length; k += 1) {
          const sourceX = Math.min(xmin + k, sourceWidth - 1);
          sum += weights[k] * input[(y * sourceWidth + sourceX) * channels + channel];
        }
        intermediate[(y * targetWidth + targetX) * channels + channel] =
          Math.min(255, Math.max(0, Math.round(sum)));
      }
    }
  }
  const output = new Float32Array(targetHeight * targetWidth * channels);
  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const { xmin, weights } = heightPlan[targetY];
    for (let x = 0; x < targetWidth; x += 1) {
      for (let channel = 0; channel < channels; channel += 1) {
        let sum = 0;
        for (let k = 0; k < weights.length; k += 1) {
          const sourceY = Math.min(xmin + k, sourceHeight - 1);
          sum += weights[k] * intermediate[(sourceY * targetWidth + x) * channels + channel];
        }
        output[(targetY * targetWidth + x) * channels + channel] =
          Math.min(255, Math.max(0, Math.round(sum)));
      }
    }
  }
  return output;
}

// ---------------------------------------------------------------------------
// Full preprocessing: decoded RGB pixels (channel-last, values 0..255) →
// smart_resize → bicubic → rescale 1/255 → normalize (x - 0.5) / 0.5 →
// patch extraction [T·gh·gw, 588] with the pinned (t, h, w) patch order and
// (channel, y, x) element order inside a patch.
export function preprocessImage(pixels, sourceHeight, sourceWidth, channels = 3) {
  if (!Number.isSafeInteger(sourceHeight) || sourceHeight <= 0 ||
      !Number.isSafeInteger(sourceWidth) || sourceWidth <= 0) {
    fail("preprocess image dimensions must be positive integers");
  }
  if (channels !== 3) {
    fail("preprocess expects RGB pixels (3 channels)");
  }
  if (pixels.length !== sourceHeight * sourceWidth * channels) {
    fail("preprocess pixel buffer length drifted");
  }
  const input = new Float32Array(pixels.length);
  for (let index = 0; index < pixels.length; index += 1) {
    const value = pixels[index];
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 255) {
      fail(`preprocess pixel ${index} is outside the 0..255 range`);
    }
    input[index] = value;
  }
  const { height, width, gridThw } = smartResize(sourceHeight, sourceWidth);
  const resized = bicubicResizeF32(input, sourceHeight, sourceWidth, height, width, channels);
  const [, gridH, gridW] = gridThw;
  const patchCount = gridThw[0] * gridH * gridW;
  const patchElements = channels * MULTIMODAL.patchSize * MULTIMODAL.patchSize;
  const pixelValues = new Float32Array(patchCount * patchElements);
  for (let time = 0; time < gridThw[0]; time += 1) {
    for (let gridY = 0; gridY < gridH; gridY += 1) {
      for (let gridX = 0; gridX < gridW; gridX += 1) {
        const patchIndex = (time * gridH + gridY) * gridW + gridX;
        for (let channel = 0; channel < channels; channel += 1) {
          for (let y = 0; y < MULTIMODAL.patchSize; y += 1) {
            for (let x = 0; x < MULTIMODAL.patchSize; x += 1) {
              const sourceY = gridY * MULTIMODAL.patchSize + y;
              const sourceX = gridX * MULTIMODAL.patchSize + x;
              const normalized =
                (resized[(sourceY * width + sourceX) * channels + channel] * MULTIMODAL.rescale -
                  MULTIMODAL.imageMean) /
                MULTIMODAL.imageStd;
              pixelValues[
                patchIndex * patchElements +
                  ((channel * MULTIMODAL.patchSize + y) * MULTIMODAL.patchSize + x)
              ] = normalized;
            }
          }
        }
      }
    }
  }
  return { pixelValues, gridThw, resizedHeight: height, resizedWidth: width };
}

// ---------------------------------------------------------------------------
// Patch embedding: the accepted patch_projection_f32 semantics (conv-14 as a
// bias linear over the 588-dim flattened patches, bias accumulator start,
// ascending input order) plus bilinear position interpolation of the
// [27×27, 1152] table (the cpu-ref bilinear_axis rule).
export function patchProjection(pixelValues, weight, bias, {
  hiddenSize = MULTIMODAL.visionHidden,
} = {}) {
  const patchElements = 3 * MULTIMODAL.patchSize * MULTIMODAL.patchSize;
  if (pixelValues.length === 0 || pixelValues.length % patchElements !== 0) {
    fail("patch projection input length drifted");
  }
  if (weight.length !== hiddenSize * patchElements) {
    fail("patch projection weight length drifted");
  }
  if (bias.length !== hiddenSize) {
    fail("patch projection bias length drifted");
  }
  const patchCount = pixelValues.length / patchElements;
  const output = new Float32Array(patchCount * hiddenSize);
  for (let patch = 0; patch < patchCount; patch += 1) {
    for (let channel = 0; channel < hiddenSize; channel += 1) {
      let accumulator = bias[channel];
      const weightBase = channel * patchElements;
      const patchBase = patch * patchElements;
      for (let depth = 0; depth < patchElements; depth += 1) {
        accumulator += pixelValues[patchBase + depth] * weight[weightBase + depth];
      }
      output[patch * hiddenSize + channel] = accumulator;
    }
  }
  return output;
}

function bilinearAxis(sourceSize, targetSize, targetIndex) {
  const coordinate = targetSize === 1
    ? 0
    : targetIndex * (sourceSize - 1) / (targetSize - 1);
  const lower = Math.min(Math.floor(coordinate), sourceSize - 1);
  const upper = Math.min(lower + 1, sourceSize - 1);
  return [lower, upper, coordinate - lower];
}

export function addPositionEmbedding(patchEmbeddings, positionEmbedding, gridThw, {
  hiddenSize = MULTIMODAL.visionHidden,
  sourceHeight = MULTIMODAL.positionGrid,
  sourceWidth = MULTIMODAL.positionGrid,
} = {}) {
  const [temporal, gridH, gridW] = gridThw;
  if (!(temporal > 0 && gridH > 0 && gridW > 0)) {
    fail("position embedding grid drifted");
  }
  const patchCount = temporal * gridH * gridW;
  if (patchEmbeddings.length !== patchCount * hiddenSize) {
    fail("position embedding input length drifted");
  }
  if (positionEmbedding.length !== sourceHeight * sourceWidth * hiddenSize) {
    fail("position embedding table length drifted");
  }
  const output = new Float32Array(patchEmbeddings);
  let tokenOffset = 0;
  for (let time = 0; time < temporal; time += 1) {
    for (let y = 0; y < gridH; y += 1) {
      const [sourceY0, sourceY1, yFraction] = bilinearAxis(sourceHeight, gridH, y);
      for (let x = 0; x < gridW; x += 1) {
        const [sourceX0, sourceX1, xFraction] = bilinearAxis(sourceWidth, gridW, x);
        const targetToken = tokenOffset + time * gridH * gridW + y * gridW + x;
        const targetStart = targetToken * hiddenSize;
        for (let channel = 0; channel < hiddenSize; channel += 1) {
          const topLeft = positionEmbedding[(sourceY0 * sourceWidth + sourceX0) * hiddenSize + channel];
          const topRight = positionEmbedding[(sourceY0 * sourceWidth + sourceX1) * hiddenSize + channel];
          const bottomLeft = positionEmbedding[(sourceY1 * sourceWidth + sourceX0) * hiddenSize + channel];
          const bottomRight = positionEmbedding[(sourceY1 * sourceWidth + sourceX1) * hiddenSize + channel];
          const top = topLeft + (topRight - topLeft) * xFraction;
          const bottom = bottomLeft + (bottomRight - bottomLeft) * xFraction;
          output[targetStart + channel] += top + (bottom - top) * yFraction;
        }
      }
    }
    tokenOffset += gridH * gridW;
  }
  return output;
}

export function patchEmbedding(pixelValues, { weight, bias, positionEmbedding, gridThw }) {
  const patch = patchProjection(pixelValues, weight, bias);
  const output = addPositionEmbedding(patch, positionEmbedding, gridThw);
  return { patch, output };
}

export async function patchEmbeddingWebGpu(
  runtime,
  pixelValues,
  { weight, bias, positionEmbedding, gridThw },
) {
  if (typeof runtime?.run_vision_patch_projection_bytes !== "function") {
    fail("WebGPU patch-projection runtime bytes API is unavailable");
  }
  const patchElements = 3 * MULTIMODAL.patchSize * MULTIMODAL.patchSize;
  if (!(pixelValues instanceof Float32Array) ||
      pixelValues.length === 0 ||
      pixelValues.length % patchElements !== 0) {
    fail("WebGPU patch-projection input length drifted");
  }
  if (!(weight instanceof Float32Array) ||
      weight.length !== MULTIMODAL.visionHidden * patchElements) {
    fail("WebGPU patch-projection weight length drifted");
  }
  if (!(bias instanceof Float32Array) || bias.length !== MULTIMODAL.visionHidden) {
    fail("WebGPU patch-projection bias length drifted");
  }
  const patchCount = pixelValues.length / patchElements;
  const execution = await runtime.run_vision_patch_projection_bytes(
    JSON.stringify({
      schema_version: 1,
      patch_count: patchCount,
      input_width: patchElements,
      output_width: MULTIMODAL.visionHidden,
      weight_storage: "f32",
    }),
    multimodalF32Bytes(pixelValues),
    multimodalF32Bytes(weight),
    multimodalF32Bytes(bias),
  );
  const checkpointBytes = execution?.checkpoint_bytes;
  const expectedBytes = patchCount * MULTIMODAL.visionHidden * 4;
  if (!(checkpointBytes instanceof Uint8Array) ||
      checkpointBytes.byteLength !== expectedBytes) {
    fail("WebGPU patch-projection output byte length drifted");
  }
  if (typeof execution.diagnostics_json !== "string") {
    fail("WebGPU patch-projection diagnostics drifted");
  }
  const patch = new Float32Array(checkpointBytes.slice().buffer);
  const output = addPositionEmbedding(patch, positionEmbedding, gridThw);
  return {
    patch,
    output,
    diagnostics: JSON.parse(execution.diagnostics_json),
  };
}

// ---------------------------------------------------------------------------
// Prompt ids: the per-case pinned text sides plus the image-run expansion
// (prod(grid) / merge^2 image tokens), exactly the golden processor layout.
export function multimodalPromptInputIds(gridThw, { prefixIds, suffixIds }) {
  const [temporal, height, width] = gridThw;
  if (!Number.isSafeInteger(temporal) || temporal <= 0 ||
      !Number.isSafeInteger(height) || height <= 0 || height % MULTIMODAL.mergeSize !== 0 ||
      !Number.isSafeInteger(width) || width <= 0 || width % MULTIMODAL.mergeSize !== 0) {
    fail("prompt grid drifted (dimensions must be positive and merge-divisible)");
  }
  if (!Array.isArray(prefixIds) || !Array.isArray(suffixIds) ||
      prefixIds.length === 0 || suffixIds.length === 0 ||
      [...prefixIds, ...suffixIds].some((id) => !Number.isSafeInteger(id) || id < 0)) {
    fail("prompt text sides drifted");
  }
  const imageTokens = temporal * (height / MULTIMODAL.mergeSize) * (width / MULTIMODAL.mergeSize);
  const ids = [...prefixIds];
  for (let index = 0; index < imageTokens; index += 1) {
    ids.push(MULTIMODAL.imageTokenId);
  }
  ids.push(...suffixIds);
  return ids;
}

// ---------------------------------------------------------------------------
// The accepted assemble_multimodal_embeddings_f32 semantics: the text rows
// are gathered from the token embedding table for every id, then the image
// rows are scattered from the projected image embeddings IN ORDER.
export function assembleEmbeddings(tokenEmbedding, projectedImageEmbeddings, inputIds, {
  imageTokenId = MULTIMODAL.imageTokenId,
  hiddenSize = MULTIMODAL.decoderHidden,
} = {}) {
  if (!Array.isArray(inputIds) || inputIds.length === 0 ||
      !inputIds.every((id) => Number.isSafeInteger(id) && id >= 0)) {
    fail("assembly input ids drifted");
  }
  if (typeof tokenEmbedding?.gather !== "function") {
    fail("assembly token embedding table must expose gather(id)");
  }
  const imageRows = inputIds.filter((id) => id === imageTokenId).length;
  if (projectedImageEmbeddings.length !== imageRows * hiddenSize) {
    fail(`assembly projected image rows drifted: expected ${imageRows}`);
  }
  const output = new Float32Array(inputIds.length * hiddenSize);
  for (let row = 0; row < inputIds.length; row += 1) {
    output.set(tokenEmbedding.gather(inputIds[row]), row * hiddenSize);
  }
  let projectedRow = 0;
  for (let row = 0; row < inputIds.length; row += 1) {
    if (inputIds[row] !== imageTokenId) continue;
    output.set(
      projectedImageEmbeddings.subarray(projectedRow * hiddenSize, (projectedRow + 1) * hiddenSize),
      row * hiddenSize,
    );
    projectedRow += 1;
  }
  if (projectedRow !== imageRows) {
    fail("assembly scatter row count drifted");
  }
  return output;
}

// ---------------------------------------------------------------------------
// The accepted mrope_position_ids semantics: text runs share one counter,
// image tokens carry [t, h, w] grid positions over the MERGED grid, the next
// text run continues at visual_start + max(h, w) + 1, and the decode
// continuation rows use rope(j + rope_delta).
export function mropePositions(inputIds, gridThw, {
  imageTokenId = MULTIMODAL.imageTokenId,
  visionStartTokenId = MULTIMODAL.visionStartTokenId,
  spatialMergeSize = MULTIMODAL.mergeSize,
} = {}) {
  if (!Array.isArray(inputIds) || inputIds.length === 0 ||
      !inputIds.every((id) => Number.isSafeInteger(id) && id >= 0)) {
    fail("mrope input ids drifted");
  }
  if (imageTokenId === visionStartTokenId) {
    fail("mrope token boundaries drifted");
  }
  const [temporal, rawHeight, rawWidth] = gridThw;
  if (!Number.isSafeInteger(temporal) || temporal <= 0 ||
      !Number.isSafeInteger(rawHeight) || rawHeight <= 0 || rawHeight % spatialMergeSize !== 0 ||
      !Number.isSafeInteger(rawWidth) || rawWidth <= 0 || rawWidth % spatialMergeSize !== 0) {
    fail("mrope grid drifted (dimensions must be positive and merge-divisible)");
  }
  const grid = {
    temporal,
    height: rawHeight / spatialMergeSize,
    width: rawWidth / spatialMergeSize,
  };
  grid.tokens = grid.temporal * grid.height * grid.width;

  const visionStart = inputIds.indexOf(visionStartTokenId);
  if (visionStart < 0) {
    fail("mrope input ids contain no vision start token");
  }
  if (inputIds.indexOf(visionStartTokenId, visionStart + 1) !== -1) {
    fail("mrope input ids contain a second vision start token");
  }
  const imageStart = visionStart + 1;
  const imageEnd = imageStart + grid.tokens;
  if (imageEnd > inputIds.length ||
      !inputIds.slice(imageStart, imageEnd).every((id) => id === imageTokenId) ||
      inputIds[imageEnd] === imageTokenId) {
    fail("mrope image run length drifted");
  }
  if (!inputIds.slice(imageEnd).every(
    (id) => id !== imageTokenId && id !== visionStartTokenId,
  )) {
    fail("mrope input ids contain image tokens outside the image run");
  }

  const positionIds = [
    new Float64Array(inputIds.length),
    new Float64Array(inputIds.length),
    new Float64Array(inputIds.length),
  ];
  for (let index = 0; index < imageStart; index += 1) {
    positionIds[0][index] = index;
    positionIds[1][index] = index;
    positionIds[2][index] = index;
  }
  const visualStart = imageStart;
  for (let time = 0; time < grid.temporal; time += 1) {
    for (let y = 0; y < grid.height; y += 1) {
      for (let x = 0; x < grid.width; x += 1) {
        const token = imageStart + (time * grid.height + y) * grid.width + x;
        positionIds[0][token] = visualStart;
        positionIds[1][token] = visualStart + y;
        positionIds[2][token] = visualStart + x;
      }
    }
  }
  const nextPosition = visualStart + Math.max(grid.height - 1, grid.width - 1) + 1;
  for (let index = imageEnd; index < inputIds.length; index += 1) {
    const position = nextPosition + (index - imageEnd);
    positionIds[0][index] = position;
    positionIds[1][index] = position;
    positionIds[2][index] = position;
  }
  let maxPosition = 0;
  for (const axis of positionIds) {
    for (const value of axis) maxPosition = Math.max(maxPosition, value);
  }
  return {
    positionIds,
    ropeDelta: maxPosition + 1 - inputIds.length,
  };
}

// ---------------------------------------------------------------------------
// Standard RoPE tables in the accepted axis-major layout [3, tokens, 128]
// (axis planes in the pinned [16, 24, 24] section layout, duplicated second
// half), with the decode-continuation rows [promptTokens, capacity) filled
// from rope(j + ropeDelta) — the same rows the M6e8 decode flow consumes.
export function ropeTables(positionIds, ropeDelta, {
  capacity = null,
  theta = MULTIMODAL.ropeTheta,
  headDim = MULTIMODAL.headDim,
} = {}) {
  if (!Array.isArray(positionIds) || positionIds.length !== 3) {
    fail("rope positions drifted");
  }
  const promptTokens = positionIds[0].length;
  if (promptTokens === 0 ||
      positionIds[1].length !== promptTokens || positionIds[2].length !== promptTokens) {
    fail("rope position axis length drifted");
  }
  if (!Number.isSafeInteger(ropeDelta)) {
    fail("rope delta drifted");
  }
  const tokens = capacity ?? promptTokens;
  if (!Number.isSafeInteger(tokens) || tokens < promptTokens) {
    fail("rope table capacity drifted");
  }
  if (typeof theta !== "number" || !Number.isFinite(theta) || theta <= 0) {
    fail("rope theta drifted");
  }
  const halfDim = headDim / 2;
  const inverseFrequencies = new Float64Array(halfDim);
  for (let index = 0; index < halfDim; index += 1) {
    inverseFrequencies[index] = theta ** (-(2 * index) / headDim);
  }
  const cos = new Float32Array(3 * tokens * headDim);
  const sin = new Float32Array(3 * tokens * headDim);
  for (let axis = 0; axis < 3; axis += 1) {
    for (let token = 0; token < tokens; token += 1) {
      const position = token < promptTokens
        ? positionIds[axis][token]
        : token + ropeDelta;
      for (let dim = 0; dim < headDim; dim += 1) {
        const half = dim >= halfDim ? dim - halfDim : dim;
        const angle = position * inverseFrequencies[half];
        const index = (axis * tokens + token) * headDim + dim;
        cos[index] = Math.cos(angle);
        sin[index] = Math.sin(angle);
      }
    }
  }
  return { cos, sin };
}

// ---------------------------------------------------------------------------
// Caller-owned detokenization for the pinned tokenizer.json (SentencePiece-
// style BPE: "▁" -> " ", <0xNN> byte fallback, added_tokens with special
// flags suppressed).
export function loadMultimodalTokenizer(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
    fail("tokenizer must be a nonempty Uint8Array");
  }
  const document = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  if (document?.model?.type !== "BPE" ||
      typeof document.model.vocab !== "object" || document.model.vocab === null ||
      !Array.isArray(document.added_tokens)) {
    fail("tokenizer schema drifted from the pinned BPE layout");
  }
  const idToToken = new Map();
  for (const [token, id] of Object.entries(document.model.vocab)) {
    if (!Number.isSafeInteger(id) || id < 0) {
      fail("tokenizer vocab id drifted");
    }
    idToToken.set(id, token);
  }
  const addedById = new Map();
  for (const added of document.added_tokens) {
    if (!Number.isSafeInteger(added.id) || added.id < 0 ||
        typeof added.content !== "string" || typeof added.special !== "boolean") {
      fail("tokenizer added_tokens entry drifted");
    }
    addedById.set(added.id, added);
  }
  const byteFallback = /^<0x[0-9A-Fa-f]{2}>$/;
  function detokenize(ids) {
    if (!Array.isArray(ids) || !ids.every((id) => Number.isSafeInteger(id) && id >= 0)) {
      fail("detokenize ids must be unsigned integers");
    }
    const output = [];
    for (const id of ids) {
      const added = addedById.get(id);
      if (added !== undefined && added.special) continue;
      const piece = added === undefined ? idToToken.get(id) : added.content;
      if (piece === undefined) {
        fail(`detokenize unknown token id ${id}`);
      }
      if (byteFallback.test(piece)) {
        output.push(Number.parseInt(piece.slice(3, 5), 16));
      } else {
        for (const byte of new TextEncoder().encode(piece.replaceAll("▁", " "))) {
          output.push(byte);
        }
      }
    }
    return new TextDecoder("utf-8").decode(new Uint8Array(output));
  }
  return Object.freeze({ idToToken, detokenize });
}

// Exact little-endian f32 widening of a row of f32 values (the prefill
// hidden operand layout).
export function multimodalF32Bytes(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setFloat32(index * 4, values[index], true);
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// Per-case decoder session pack builder (PVLCPK01, the accepted M6e6/M6e8
// layout authority mirrored in the runner layer): the 11 accepted decoder
// weight shards plus `weights.final_layernorm` and `weights.lm_head` in the
// pinned order, canonical-JSON manifest/descriptor, 56-byte fixed directory
// entries, 256-byte section alignment. The demo and the browser page build
// per-case packs in-page because the rope tables and the cache capacity are
// per-case pipeline outputs; the decoder weights are the accepted official
// payloads. `blake3Hex` is the digest authority (the WASM runtime's
// blake3_bytes_hex in the browser).
const SESSION_PACK_SHARD_IDS = Object.freeze([
  "weights.input_layernorm",
  "weights.q_proj",
  "weights.k_proj",
  "weights.v_proj",
  "weights.o_proj",
  "weights.mrope_cos",
  "weights.mrope_sin",
  "weights.post_attention_layernorm",
  "weights.gate_proj",
  "weights.up_proj",
  "weights.down_proj",
  "weights.final_layernorm",
  "weights.lm_head",
]);

const SESSION_PACK_SHARD_ROLES = Object.freeze({
  "weights.input_layernorm": "norm1",
  "weights.q_proj": "q",
  "weights.k_proj": "k",
  "weights.v_proj": "v",
  "weights.o_proj": "o",
  "weights.mrope_cos": "ropeCos",
  "weights.mrope_sin": "ropeSin",
  "weights.post_attention_layernorm": "norm2",
  "weights.gate_proj": "gate",
  "weights.up_proj": "up",
  "weights.down_proj": "down",
  "weights.final_layernorm": "finalNorm",
  "weights.lm_head": "lmHead",
});

const SESSION_PACK_MODEL_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const SESSION_PACK_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6";
const SESSION_PACK_HEADER_BYTES = 32;
const SESSION_PACK_DIRECTORY_FIXED_BYTES = 56;
const SESSION_PACK_SECTION_ALIGNMENT = 256;

function sessionPackCanonicalJson(value, formatNumber = JSON.stringify) {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      fail("session pack JSON number is not finite");
    }
    return formatNumber(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => sessionPackCanonicalJson(item, formatNumber)).join(",")}]`;
  }
  if (typeof value !== "object") {
    fail("session pack JSON value is unsupported");
  }
  const entries = Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${sessionPackCanonicalJson(value[key], formatNumber)}`);
  return `{${entries.join(",")}}`;
}

function sessionPackCanonicalJsonBytes(value, formatNumber) {
  return new TextEncoder().encode(`${sessionPackCanonicalJson(value, formatNumber)}\n`);
}

function sessionPackAlignUp(value, alignment) {
  const remainder = value % alignment;
  return remainder === 0 ? value : value + alignment - remainder;
}

function sessionPackDigestBytes(hex) {
  const bytes = new Uint8Array(32);
  for (let index = 0; index < 32; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

export function buildMultimodalSessionPack({
  descriptor,
  weights,
  blake3Hex,
  oracle = "official_l3",
  caseId = "official.decoder_stack_multimodal_demo",
  weightStorage = "f32",
  checkpointIdentity = null,
}) {
  if (typeof blake3Hex !== "function") {
    fail("session pack builder needs a BLAKE3 digest function");
  }
  if (oracle !== "synthetic" && oracle !== "official_l3") {
    fail("unknown session pack oracle");
  }
  if (typeof caseId !== "string" || caseId.length === 0) {
    fail("invalid session pack case id");
  }
  if (!Number.isSafeInteger(descriptor.prefix_tokens) || descriptor.prefix_tokens < 0 ||
      !Number.isSafeInteger(descriptor.cache_capacity) ||
      descriptor.cache_capacity <= descriptor.prefix_tokens) {
    fail("session pack descriptor cache geometry drifted");
  }
  if (weightStorage !== "f32" && weightStorage !== "f16") {
    fail("session pack weight storage is unsupported");
  }
  if (weightStorage === "f16" &&
      (checkpointIdentity === null ||
       typeof checkpointIdentity !== "object" ||
       typeof checkpointIdentity.blake3 !== "string" ||
       !/^[0-9a-f]{64}$/.test(checkpointIdentity.blake3) ||
       !Number.isSafeInteger(checkpointIdentity.bytes) ||
       checkpointIdentity.bytes <= 0)) {
    fail("F16 session pack checkpoint identity is invalid");
  }
  const payloadEntries = SESSION_PACK_SHARD_IDS.map((id) => {
    const values = weights[SESSION_PACK_SHARD_ROLES[id]];
    const ropeTable = id === "weights.mrope_cos" || id === "weights.mrope_sin";
    const expectedF32 = weightStorage === "f32" || ropeTable;
    if (expectedF32 && !(values instanceof Float32Array)) {
      fail(`session pack shard ${id} operand is missing or not Float32Array`);
    }
    if (!expectedF32 &&
        (!(values instanceof Uint8Array) || values.byteLength === 0 ||
         values.byteLength % 2 !== 0)) {
      fail(`session pack shard ${id} operand is missing or not aligned F16 bytes`);
    }
    const payload = values instanceof Float32Array
      ? new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
      : values;
    return {
      id,
      kind: 2,
      payload,
      blake3: blake3Hex(payload),
      dtype: expectedF32 ? "f32" : "f16",
    };
  });
  const descriptorObject = {
    schema_version: 1,
    oracle,
    case_id: caseId,
    model_revision: SESSION_PACK_MODEL_REVISION,
    layers: 18,
    hidden_size: 1024,
    intermediate_size: 3072,
    query_heads: 16,
    key_value_heads: 2,
    head_dim: 128,
    query_width: 2048,
    key_value_width: 256,
    prefix_tokens: descriptor.prefix_tokens,
    cache_capacity: descriptor.cache_capacity,
    rms_norm_epsilon: 0.00001,
    mrope_sections: [16, 24, 24],
    ...(weightStorage === "f16"
      ? {
          checkpoint_blake3: checkpointIdentity.blake3,
          checkpoint_bytes: checkpointIdentity.bytes,
          weight_storage: "f16",
        }
      : {}),
    shards: Object.fromEntries(
      payloadEntries.map((entry) => [
        entry.id,
        {
          bytes: entry.payload.byteLength,
          blake3: entry.blake3,
          ...(weightStorage === "f16" ? { dtype: entry.dtype } : {}),
        },
      ]),
    ),
  };
  const descriptorPayload = sessionPackCanonicalJsonBytes(descriptorObject);
  const manifestPayload = sessionPackCanonicalJsonBytes({
    compiler_build: "0".repeat(64),
    compiler_model_abi: 1,
    context_limit: 4096,
    model_id: SESSION_PACK_MODEL_ID,
    model_revision: SESSION_PACK_MODEL_REVISION,
    precision_profile: weightStorage === "f16" ? "balanced" : "fidelity",
    resolution_buckets: [[672, 672]],
  });
  const sections = [
    {
      id: "ir.decoder_stack_00",
      kind: 1,
      payload: descriptorPayload,
      blake3: blake3Hex(descriptorPayload),
    },
    ...payloadEntries,
  ];
  const directoryEntries = sections.map((section) => {
    const idBytes = new TextEncoder().encode(section.id);
    const entryBytes = sessionPackAlignUp(
      SESSION_PACK_DIRECTORY_FIXED_BYTES + idBytes.length,
      8,
    );
    return { ...section, idBytes, entryBytes };
  });
  const directoryLength = directoryEntries.reduce(
    (total, entry) => total + entry.entryBytes,
    0,
  );
  let cursor = SESSION_PACK_HEADER_BYTES + manifestPayload.byteLength + directoryLength;
  for (const entry of directoryEntries) {
    cursor = sessionPackAlignUp(cursor, SESSION_PACK_SECTION_ALIGNMENT);
    entry.offset = cursor;
    cursor += entry.payload.byteLength;
  }
  const fileLength = cursor;
  const pack = new Uint8Array(fileLength);
  const view = new DataView(pack.buffer);
  pack.set(new TextEncoder().encode("PVLCPK01"), 0);
  view.setUint32(8, 1, true);
  view.setUint32(12, manifestPayload.byteLength, true);
  view.setUint32(16, directoryLength, true);
  view.setUint32(20, directoryEntries.length, true);
  view.setBigUint64(24, BigInt(fileLength), true);
  pack.set(manifestPayload, SESSION_PACK_HEADER_BYTES);
  let directoryCursor = SESSION_PACK_HEADER_BYTES + manifestPayload.byteLength;
  for (const entry of directoryEntries) {
    view.setUint16(directoryCursor, entry.idBytes.length, true);
    view.setUint8(directoryCursor + 2, entry.kind);
    view.setUint8(directoryCursor + 3, 0);
    view.setUint32(directoryCursor + 4, SESSION_PACK_SECTION_ALIGNMENT, true);
    view.setBigUint64(directoryCursor + 8, BigInt(entry.offset), true);
    view.setBigUint64(directoryCursor + 16, BigInt(entry.payload.byteLength), true);
    pack.set(sessionPackDigestBytes(entry.blake3), directoryCursor + 24);
    pack.set(entry.idBytes, directoryCursor + SESSION_PACK_DIRECTORY_FIXED_BYTES);
    directoryCursor += entry.entryBytes;
    pack.set(entry.payload, entry.offset);
  }
  return pack;
}

// ---------------------------------------------------------------------------
// Minimal PVLCPK01 shard reader for the accepted decoder session packs
// (their shard order is the pinned semantic order, NOT the canonical sorted
// order the projector pack family uses): walks the directory, maps shard id
// to its payload bytes, and verifies every directory digest against the
// payload via the caller-provided BLAKE3 function.
export function readSessionPackShardBytes(packBytes, blake3Hex, shardIds) {
  if (!(packBytes instanceof Uint8Array)) {
    fail("session pack must be Uint8Array");
  }
  if (typeof blake3Hex !== "function") {
    fail("session pack reader needs a BLAKE3 digest function");
  }
  const view = new DataView(packBytes.buffer, packBytes.byteOffset, packBytes.byteLength);
  if (new TextDecoder().decode(packBytes.subarray(0, 8)) !== "PVLCPK01") {
    fail("session pack magic mismatch");
  }
  if (view.getUint32(8, true) !== 1) {
    fail("session pack version is unsupported");
  }
  const manifestLength = view.getUint32(12, true);
  const directoryLength = view.getUint32(16, true);
  const sectionCount = view.getUint32(20, true);
  const fileLength = view.getBigUint64(24, true);
  if (fileLength !== BigInt(packBytes.byteLength)) {
    fail("session pack declared length drifted");
  }
  const decoder = new TextDecoder();
  const shards = new Map();
  let cursor = SESSION_PACK_HEADER_BYTES + manifestLength;
  const directoryEnd = cursor + directoryLength;
  for (let index = 0; index < sectionCount; index += 1) {
    const idLength = view.getUint16(cursor, true);
    const offset = Number(view.getBigUint64(cursor + 8, true));
    const byteLength = Number(view.getBigUint64(cursor + 16, true));
    const id = decoder.decode(packBytes.subarray(cursor + 56, cursor + 56 + idLength));
    const digest = [...packBytes.subarray(cursor + 24, cursor + 56)]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
    const payload = packBytes.subarray(offset, offset + byteLength);
    if (blake3Hex(payload) !== digest) {
      fail(`session pack shard ${id} directory digest mismatch`);
    }
    shards.set(id, payload);
    cursor += sessionPackAlignUp(SESSION_PACK_DIRECTORY_FIXED_BYTES + idLength, 8);
  }
  if (cursor !== directoryEnd) {
    fail("session pack directory length drifted");
  }
  const wanted = shardIds ?? SESSION_PACK_SHARD_IDS;
  for (const id of wanted) {
    if (!shards.has(id)) {
      fail(`session pack shard ${id} is missing`);
    }
  }
  return shards;
}

export function readSessionPackShards(packBytes, blake3Hex, shardIds) {
  const shards = readSessionPackShardBytes(packBytes, blake3Hex, shardIds);
  const wanted = shardIds ?? SESSION_PACK_SHARD_IDS;
  const floats = new Map();
  for (const id of wanted) {
    const payload = shards.get(id);
    floats.set(
      id,
      new Float32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4),
    );
  }
  return floats;
}

// ---------------------------------------------------------------------------
// Arbitrary-input admission (M6e11). The accepted M3 vision stack session
// and the M4 projector session already admit caller-declared
// `synthetic`-oracle manifests/packs: provenance-free, structurally
// validated shard directories (crates/pvlc-pack/src/vision_stack_shards.rs —
// Synthetic rejects only official provenance fields, the shard directory is
// validated for id/kind/order/bytes/blake3 form; and
// crates/pvlc-pack/src/projector_self_test.rs — the synthetic projector
// descriptor rejects official provenance and official profile names). The
// builders below declare exactly that accepted surface with the official
// weights and the caller-computed inputs. The sealed official
// manifests/packs keep their accepted provenance paths unchanged.

// A `synthetic`-oracle vision-stack manifest for a caller-computed input:
// tokens and cu_seqlens from the image grid, checkpoint layers [0, 1, 13,
// 26], the pinned vision geometry (1152/16/72/4304/eps 1e-6, 27 layers), and
// NO provenance fields (golden_bundle_digest/semantic_fingerprint must be
// ABSENT — the synthetic oracle rejects their presence). `weightShards` are
// the accepted official weight shard pins ({id, bytes, blake3} copied from
// the locked official manifest); the input.embeddings pin is declared from
// the caller-computed patch embeddings.
export function buildVisionStackManifest({
  tokens,
  weightShards,
  inputBlake3,
  caseId = "multimodal.arbitrary_vision_stack",
  matrixWeightStorage = "f32",
  matrixWeightLayout = "output_major",
  vectorWeightStorage = "f32",
  activationStorage = "f32",
  checkpointLayers = [0, 1, 13, 26],
}) {
  if (!Number.isSafeInteger(tokens) || tokens <= 0) {
    fail("vision-stack synthetic manifest token count drifted");
  }
  if (typeof caseId !== "string" || caseId.length === 0 || caseId.length > 256 ||
      !/^[a-zA-Z0-9._/-]+$/.test(caseId)) {
    fail("vision-stack synthetic manifest case id is invalid");
  }
  if (!Array.isArray(weightShards) || weightShards.length !== 28) {
    fail("vision-stack synthetic manifest weight shard count drifted");
  }
  if (typeof inputBlake3 !== "string" || !/^[0-9a-f]{64}$/.test(inputBlake3)) {
    fail("vision-stack synthetic manifest input digest is invalid");
  }
  if (matrixWeightStorage !== "f32" && matrixWeightStorage !== "f16") {
    fail("vision-stack matrix weight storage is invalid");
  }
  if (matrixWeightLayout !== "output_major" && matrixWeightLayout !== "input_major") {
    fail("vision-stack matrix weight layout is invalid");
  }
  if (matrixWeightLayout === "input_major" && matrixWeightStorage !== "f16") {
    fail("vision-stack input-major matrix weight layout requires FP16 storage");
  }
  if (vectorWeightStorage !== "f32" && vectorWeightStorage !== "f16") {
    fail("vision-stack vector weight storage is invalid");
  }
  if (activationStorage !== "f32" && activationStorage !== "f16") {
    fail("vision-stack activation storage is invalid");
  }
  const legacyF32 =
    vectorWeightStorage === "f32" && activationStorage === "f32";
  const fullF16 =
    matrixWeightStorage === "f16" &&
    matrixWeightLayout === "input_major" &&
    vectorWeightStorage === "f16" &&
    activationStorage === "f16";
  if (!legacyF32 && !fullF16) {
    fail("vision-stack precision profile is incoherent");
  }
  if (
    !Array.isArray(checkpointLayers) ||
    checkpointLayers.some(
      (layer, index) =>
        !Number.isSafeInteger(layer) ||
        layer < 0 ||
        layer >= 27 ||
        (index > 0 && checkpointLayers[index - 1] >= layer),
    )
  ) {
    fail("vision-stack checkpoint layer selection is invalid");
  }
  const activationBytes = activationStorage === "f16" ? 2 : 4;
  const shards = [
    {
      id: "input.embeddings",
      kind: "input",
      layer_index: null,
      bytes: tokens * 1152 * activationBytes,
      blake3: inputBlake3,
    },
    ...weightShards.map((shard) => ({
      id: shard.id,
      kind: shard.kind,
      layer_index: shard.layer_index,
      bytes: shard.bytes,
      blake3: shard.blake3,
    })),
  ];
  if (shards.length !== 29 ||
      shards[1].id !== "weights.vision_layer.00" ||
      shards[27].id !== "weights.vision_layer.26" ||
      shards[28].id !== "weights.vision_post_norm") {
    fail("vision-stack synthetic manifest shard directory drifted");
  }
  return {
    schema_version: 1,
    oracle: "synthetic",
    case_id: caseId,
    model_id: "PaddlePaddle/PaddleOCR-VL-1.6",
    model_revision: "66317acc4c9fc17bd154591ce650735cd2855f3e",
    compiler_model_abi: 1,
    compiler_build: "0".repeat(64),
    golden_bundle_digest: null,
    semantic_fingerprint: null,
    ...(matrixWeightStorage === "f16"
      ? { matrix_weight_storage: "f16" }
      : {}),
    ...(matrixWeightLayout === "input_major"
      ? { matrix_weight_layout: "input_major" }
      : {}),
    ...(vectorWeightStorage === "f16"
      ? { vector_weight_storage: "f16" }
      : {}),
    ...(activationStorage === "f16"
      ? { activation_storage: "f16" }
      : {}),
    tokens,
    hidden_size: 1152,
    attention_heads: 16,
    head_dim: 72,
    intermediate_size: 4304,
    layer_norm_epsilon: 0.000001,
    cu_seqlens: [0, tokens],
    layer_count: 27,
    checkpoint_layers: [...checkpointLayers],
    shards,
  };
}

// Generic PVLCPK01 section writer for the projector self-test pack family:
// sections are laid out in the canonical strictly-increasing id order (the
// projector pack family requires it), 56-byte fixed directory entries,
// 256-byte section alignment, each payload digest declared in its directory
// entry.
export function buildCanonicalPvlcPack({ manifestObject, sections }) {
  const sorted = [...sections].sort((left, right) => left.id.localeCompare(right.id));
  for (let index = 1; index < sorted.length; index += 1) {
    if (sorted[index - 1].id >= sorted[index].id) {
      fail("canonical pack section ids are not strictly increasing");
    }
  }
  const manifestPayload = sessionPackCanonicalJsonBytes(manifestObject);
  const directoryEntries = sorted.map((section) => {
    const idBytes = new TextEncoder().encode(section.id);
    const entryBytes = sessionPackAlignUp(
      SESSION_PACK_DIRECTORY_FIXED_BYTES + idBytes.length,
      8,
    );
    return { ...section, idBytes, entryBytes };
  });
  const directoryLength = directoryEntries.reduce(
    (total, entry) => total + entry.entryBytes,
    0,
  );
  let cursor = SESSION_PACK_HEADER_BYTES + manifestPayload.byteLength + directoryLength;
  for (const entry of directoryEntries) {
    cursor = sessionPackAlignUp(cursor, entry.alignment ?? SESSION_PACK_SECTION_ALIGNMENT);
    entry.offset = cursor;
    cursor += entry.bytes.byteLength;
  }
  const fileLength = cursor;
  const pack = new Uint8Array(fileLength);
  const view = new DataView(pack.buffer);
  pack.set(new TextEncoder().encode("PVLCPK01"), 0);
  view.setUint32(8, 1, true);
  view.setUint32(12, manifestPayload.byteLength, true);
  view.setUint32(16, directoryLength, true);
  view.setUint32(20, directoryEntries.length, true);
  view.setBigUint64(24, BigInt(fileLength), true);
  pack.set(manifestPayload, SESSION_PACK_HEADER_BYTES);
  let directoryCursor = SESSION_PACK_HEADER_BYTES + manifestPayload.byteLength;
  for (const entry of directoryEntries) {
    view.setUint16(directoryCursor, entry.idBytes.length, true);
    view.setUint8(directoryCursor + 2, entry.kind);
    view.setUint8(directoryCursor + 3, 0);
    view.setUint32(directoryCursor + 4, entry.alignment ?? SESSION_PACK_SECTION_ALIGNMENT, true);
    view.setBigUint64(directoryCursor + 8, BigInt(entry.offset), true);
    view.setBigUint64(directoryCursor + 16, BigInt(entry.bytes.byteLength), true);
    pack.set(sessionPackDigestBytes(entry.blake3), directoryCursor + 24);
    pack.set(entry.idBytes, directoryCursor + SESSION_PACK_DIRECTORY_FIXED_BYTES);
    directoryCursor += entry.entryBytes;
    pack.set(entry.bytes, entry.offset);
  }
  return pack;
}

// A `synthetic`-oracle projector self-test pack for a caller-computed input:
// the accepted official projector weights (with their {bytes, blake3} pin),
// a caller-declared input section (the computed vision.final bytes), and a
// structurally valid expected section (the self-test checkpoint surface; the
// runtime requires only its exact byte length — for the pinned golden case
// it carries the golden projector.final, for arbitrary images it is a
// caller-owned placeholder of the same length). The descriptor carries NO
// provenance and a non-official profile name (the synthetic oracle rejects
// both).
// The pinned projector LayerNorm epsilon (fround(1e-5)). The canonical
// descriptor byte form is produced by serde_json re-serialization, which
// prints this f64 in ryu scientific form ("9.999999747378752e-6") while
// JSON.stringify prints the decimal form — the formatter below bridges that
// one constant (all other descriptor numbers are integers).
const PROJECTOR_LAYER_NORM_EPSILON = Math.fround(0.00001);
const PROJECTOR_LAYER_NORM_EPSILON_JSON = "9.999999747378752e-6";

function projectorDescriptorNumberJson(value) {
  return value === PROJECTOR_LAYER_NORM_EPSILON
    ? PROJECTOR_LAYER_NORM_EPSILON_JSON
    : JSON.stringify(value);
}

export function buildSyntheticProjectorPack({
  weightsBytes,
  weightsBlake3,
  profile,
  imageGridThw,
  inputBytes,
  expectedBytes,
  blake3Hex,
}) {
  if (!(weightsBytes instanceof Uint8Array) ||
      typeof weightsBlake3 !== "string" || !/^[0-9a-f]{64}$/.test(weightsBlake3)) {
    fail("synthetic projector weights pin drifted");
  }
  if (typeof profile !== "string" ||
      profile === "ocr-clean-latin-l3" || profile === "table-simple-l2" ||
      !/^[a-zA-Z0-9._/-]+$/.test(profile)) {
    fail("synthetic projector profile name drifted");
  }
  if (!(inputBytes instanceof Uint8Array) || !(expectedBytes instanceof Uint8Array)) {
    fail("synthetic projector input/expected sections must be Uint8Array");
  }
  if (typeof blake3Hex !== "function") {
    fail("synthetic projector pack builder needs a BLAKE3 digest function");
  }
  const grid = imageGridThw[0];
  const tokens = grid[0] * grid[1] * grid[2];
  const outputTokens = (grid[1] / 2) * (grid[2] / 2) * grid[0];
  if (inputBytes.byteLength !== tokens * 1152 * 4) {
    fail("synthetic projector input byte length drifted");
  }
  if (expectedBytes.byteLength !== outputTokens * 1024 * 4) {
    fail("synthetic projector expected byte length drifted");
  }
  const descriptor = {
    schema_version: 1,
    oracle: "synthetic",
    model_revision: "66317acc4c9fc17bd154591ce650735cd2855f3e",
    hidden_size: 1152,
    output_size: 1024,
    // The pinned projector epsilon, bit-exact with the official descriptor
    // (fround(1e-5) = 0.000009999999747378752 — the sealed compute path
    // reads eps from the descriptor, so the bit-exact value is required).
    layer_norm_epsilon: PROJECTOR_LAYER_NORM_EPSILON,
    weights: {
      section_id: "weights.projector",
      bytes: weightsBytes.byteLength,
      blake3: weightsBlake3,
    },
    cases: [
      {
        profile,
        case_id: `multimodal.${profile}/projector`,
        trace_level: "synthetic",
        golden_bundle_digest: null,
        semantic_fingerprint: null,
        image_grid_thw: [grid],
        readback: "output_only",
        input: {
          section_id: `input.projector.${profile}`,
          bytes: inputBytes.byteLength,
          blake3: blake3Hex(inputBytes),
        },
        expected: {
          section_id: `self_test.projector.${profile}`,
          bytes: expectedBytes.byteLength,
          blake3: blake3Hex(expectedBytes),
        },
        stage_order: ["linear2"],
      },
    ],
  };
  const descriptorBytes = sessionPackCanonicalJsonBytes(descriptor, projectorDescriptorNumberJson);
  const pack = buildCanonicalPvlcPack({
    manifestObject: {
      compiler_build: "0".repeat(64),
      compiler_model_abi: 1,
      context_limit: 4096,
      model_id: "PaddlePaddle/PaddleOCR-VL-1.6",
      model_revision: "66317acc4c9fc17bd154591ce650735cd2855f3e",
      precision_profile: "fidelity",
      resolution_buckets: [[grid[1], grid[2]]],
    },
    sections: [
      {
        id: "ir.projector.official",
        kind: 1,
        alignment: 64,
        bytes: descriptorBytes,
        blake3: blake3Hex(descriptorBytes),
      },
      {
        id: `input.projector.${profile}`,
        kind: 2,
        bytes: inputBytes,
        blake3: blake3Hex(inputBytes),
      },
      {
        id: `self_test.projector.${profile}`,
        kind: 3,
        bytes: expectedBytes,
        blake3: blake3Hex(expectedBytes),
      },
      {
        id: "weights.projector",
        kind: 2,
        bytes: weightsBytes,
        blake3: weightsBlake3,
      },
    ],
  });
  return { packBytes: pack, descriptorJson: new TextDecoder().decode(descriptorBytes) };
}

// The canonical byte form of a vision-stack manifest (minified JSON with a
// trailing newline, in the pinned field order — the runtime re-serializes
// and compares byte-exactly, matching the locked official manifest format).
export function visionStackManifestJson(manifest) {
  return `${JSON.stringify(manifest)}\n`;
}
