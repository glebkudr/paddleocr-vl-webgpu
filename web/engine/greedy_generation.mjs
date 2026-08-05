// PVLC M6e9 caller-owned greedy generation driver
// (docs/m6e9_browser_greedy_generation_contract.md).
//
// A pure caller-owned composition of the accepted persistent-session
// operations (prefill/step/logits) into a greedy generation loop. The module
// is self-contained within the runner layer: it imports NOTHING from
// web/tests — the M6c1 top-1 comparator (value descending under the f32
// total order, ties broken by the smaller token id) is implemented locally.
// The driver holds no runtime state, never mutates its inputs, and surfaces
// session rejections exactly as the runtime raises them (no new failure
// modes).

const DEFAULT_HIDDEN_SIZE = 1024;
const LOGITS_BYTES = 103424 * 4;

function fail(message) {
  throw new Error(`greedy generation: ${message}`);
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Exact little-endian f32 widening of one f32 row (the step operand: the
// gathered embedding row).
export function f32RowToBytes(row) {
  const bytes = new Uint8Array(row.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < row.length; index += 1) {
    view.setFloat32(index * 4, row[index], true);
  }
  return bytes;
}

// The exact M6c1 top-1 comparator over the f32 total order: larger value
// first; -0.0 ranks below +0.0; +NaN above everything, -NaN below everything
// (unreachable on the contract's finite logits, kept so the comparator is a
// faithful total_cmp mirror); exact bit ties are broken by the smaller
// token id.
const TOP1_F32_SCRATCH = new Float32Array(1);
const TOP1_I32_SCRATCH = new Int32Array(TOP1_F32_SCRATCH.buffer);

function f32TotalOrderKey(value) {
  TOP1_F32_SCRATCH[0] = value;
  const bits = TOP1_I32_SCRATCH[0];
  return (bits ^ ((bits >> 31) >>> 1)) | 0;
}

export function compareGreedyTopKEntries(left, right) {
  const leftKey = f32TotalOrderKey(left.value);
  const rightKey = f32TotalOrderKey(right.value);
  if (leftKey !== rightKey) return leftKey > rightKey ? -1 : 1;
  return left.index - right.index;
}

// Top-1 selection over the raw little-endian f32 logits readback, exactly
// the accepted M6c1 comparator. The readback is scanned once, directly.
export function top1FromLogitsBytes(logitsBytes) {
  if (!(logitsBytes instanceof Uint8Array) || logitsBytes.byteLength !== LOGITS_BYTES) {
    fail(`logits readback must be a ${LOGITS_BYTES}-byte Uint8Array`);
  }
  const view = new DataView(logitsBytes.buffer, logitsBytes.byteOffset, logitsBytes.byteLength);
  let bestIndex = -1;
  let bestValue = 0;
  for (let index = 0; index < LOGITS_BYTES / 4; index += 1) {
    const value = view.getFloat32(index * 4, true);
    if (!Number.isFinite(value)) {
      fail(`logits readback contains a nonfinite f32 at ${index}`);
    }
    if (
      bestIndex < 0 ||
      compareGreedyTopKEntries({ index, value }, { index: bestIndex, value: bestValue }) < 0
    ) {
      bestIndex = index;
      bestValue = value;
    }
  }
  return { index: bestIndex, value: bestValue };
}

// One greedy generation loop over a LIVE M6e8 logits-capable session (the
// prompt was already admitted by the accepted prefill). Every iteration:
// the capacity check from the tracked cache token count (BEFORE any call —
// a zero-step capacity admission yields an empty token id sequence), then
// the accepted pure-readout logits, the exact top-1 selection, the EOS
// stop (no step), otherwise the accepted decode step on the gathered
// embedding row. `embedding` is either a Float32Array [vocab, hidden] table
// or an object with gather(id); it is never mutated. `cacheCapacity` is the
// session's admitted capacity and `initialCacheTokens` the cache token
// count after the prompt admission (the prefill token count, or the caller's
// last accepted step count); later counts are tracked from the runtime's
// own step diagnostics.
export async function runGreedyGeneration(runtime, {
  embedding,
  eosTokenId,
  maxSteps,
  cacheCapacity,
  initialCacheTokens,
  onStep = null,
}) {
  if (embedding === null || embedding === undefined) {
    fail("embedding table is missing");
  }
  const gather = embedding instanceof Float32Array
    ? (id) => embedding.subarray(id * DEFAULT_HIDDEN_SIZE, (id + 1) * DEFAULT_HIDDEN_SIZE)
    : embedding.gather?.bind(embedding);
  if (typeof gather !== "function") {
    fail("embedding must be a Float32Array table or expose gather(id)");
  }
  if (!Number.isSafeInteger(eosTokenId) || eosTokenId < 0) {
    fail("eosTokenId must be an unsigned integer");
  }
  if (!Number.isSafeInteger(maxSteps) || maxSteps <= 0) {
    fail("maxSteps must be a positive integer");
  }
  if (!Number.isSafeInteger(cacheCapacity) || cacheCapacity <= 0) {
    fail("cacheCapacity must be a positive integer");
  }
  if (
    !Number.isSafeInteger(initialCacheTokens) ||
    initialCacheTokens < 0 ||
    initialCacheTokens > cacheCapacity
  ) {
    fail("initialCacheTokens must satisfy 0 <= tokens <= cacheCapacity");
  }
  if (onStep !== null && typeof onStep !== "function") {
    fail("onStep must be a function when provided");
  }

  const tokenIds = [];
  const steps = [];
  const gpuTop1 = typeof runtime.top1_decoder_stack_session === "function";
  let logitsReadbackBytes = 0;
  let selectionQueueWallTimeNs = 0;
  let cacheTokens = initialCacheTokens;
  let stopReason = "max_steps";
  for (let iteration = 0; iteration < maxSteps; iteration += 1) {
    if (cacheTokens >= cacheCapacity) {
      stopReason = "capacity_exhausted";
      break;
    }
    const selectionResult = gpuTop1
      ? await runtime.top1_decoder_stack_session()
      : await runtime.logits_decoder_stack_session();
    const diagnostics = JSON.parse(selectionResult.diagnostics_json);
    if (
      !isRecord(diagnostics) ||
      diagnostics.cache_tokens !== cacheTokens
    ) {
      fail(
        `logits diagnostics cache_tokens drifted from the tracked ${cacheTokens}`,
      );
    }
    const logitsBytes = gpuTop1 ? null : selectionResult.logits_bytes;
    const top1 = gpuTop1
      ? { index: selectionResult.token_id, value: selectionResult.value }
      : top1FromLogitsBytes(logitsBytes);
    if (
      !Number.isSafeInteger(top1.index) ||
      top1.index < 0 ||
      top1.index >= LOGITS_BYTES / 4 ||
      !Number.isFinite(top1.value)
    ) {
      fail("GPU top-1 returned an invalid token id or score");
    }
    logitsReadbackBytes += diagnostics.readback_bytes ?? logitsBytes?.byteLength ?? 0;
    selectionQueueWallTimeNs += diagnostics.queue_wall_time_ns ?? 0;
    const position = cacheTokens;
    tokenIds.push(top1.index);
    steps.push(Object.freeze({
      position,
      selectedTokenId: top1.index,
      value: top1.value,
    }));
    if (onStep !== null) {
      await onStep(Object.freeze({
        position,
        selectedTokenId: top1.index,
        logitsBytes,
        gpuTop1,
      }));
    }
    if (top1.index === eosTokenId) {
      stopReason = "eos";
      break;
    }
    const stepResult = await runtime.step_decoder_stack_session(
      f32RowToBytes(gather(top1.index)),
    );
    const stepDiagnostics = JSON.parse(stepResult.diagnostics_json);
    if (
      !isRecord(stepDiagnostics) ||
      stepDiagnostics.cache_tokens_after !== cacheTokens + 1
    ) {
      fail("step diagnostics cache_tokens transition drifted");
    }
    cacheTokens += 1;
  }
  return Object.freeze({
    tokenIds: Object.freeze([...tokenIds]),
    stopReason,
    steps: Object.freeze(steps),
    cacheTokensAfter: cacheTokens,
    gpuTop1,
    logitsReadbackBytes,
    selectionQueueWallTimeNs,
  });
}
