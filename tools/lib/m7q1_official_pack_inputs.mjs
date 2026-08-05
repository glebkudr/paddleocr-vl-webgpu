import {
  buildM7q1BalancedWeightPack,
  M7Q1_LOGITS_WEIGHT_SHARD_IDS,
} from "./m7q1_balanced_pack.mjs";
import { checkpointIdentity } from "./m7q1_checkpoint_identity.mjs";
import {
  M6E7_OFFICIAL_CAPACITY,
  M6E7_OFFICIAL_HEAD_DIM,
  M6E7_OFFICIAL_PATHS,
  M6E7_OFFICIAL_TOKENS,
  readOfficialBf16Tensor,
  readOfficialF16Tensor,
} from "../../web/tests/support/m6e7_official_prefill_case.mjs";

const LAYERS = 18;
const LAYER_WEIGHT_ROLES = Object.freeze([
  ["weights.input_layernorm", "input_layernorm.weight", [1024]],
  ["weights.q_proj", "self_attn.q_proj.weight", [2048, 1024]],
  ["weights.k_proj", "self_attn.k_proj.weight", [256, 1024]],
  ["weights.v_proj", "self_attn.v_proj.weight", [256, 1024]],
  ["weights.o_proj", "self_attn.o_proj.weight", [1024, 2048]],
  ["weights.post_attention_layernorm", "post_attention_layernorm.weight", [1024]],
  ["weights.gate_proj", "mlp.gate_proj.weight", [3072, 1024]],
  ["weights.up_proj", "mlp.up_proj.weight", [3072, 1024]],
  ["weights.down_proj", "mlp.down_proj.weight", [1024, 3072]],
]);

function invariant(condition, message) {
  if (!condition) throw new Error(`M7q1 official pack inputs: ${message}`);
}

function concatenate(parts, label) {
  invariant(
    parts.every((part) => part instanceof Uint8Array),
    `${label} reader returned a non-byte payload`,
  );
  const byteLength = parts.reduce((total, part) => total + part.byteLength, 0);
  const combined = new Uint8Array(byteLength);
  let offset = 0;
  for (const part of parts) {
    combined.set(part, offset);
    offset += part.byteLength;
  }
  return combined;
}

export function loadM7q1OfficialRopeTables(root) {
  const prefillCos = readOfficialBf16Tensor(
    root,
    M6E7_OFFICIAL_PATHS.stackFixture,
    "decoder.rope.cos.axis_major",
    [3, M6E7_OFFICIAL_TOKENS, M6E7_OFFICIAL_HEAD_DIM],
  ).values;
  const prefillSin = readOfficialBf16Tensor(
    root,
    M6E7_OFFICIAL_PATHS.stackFixture,
    "decoder.rope.sin.axis_major",
    [3, M6E7_OFFICIAL_TOKENS, M6E7_OFFICIAL_HEAD_DIM],
  ).values;
  const decodeCos = readOfficialBf16Tensor(
    root,
    M6E7_OFFICIAL_PATHS.decodeFixture,
    "decoder.decode.00.rope.cos.axis_major",
    [3, 1, M6E7_OFFICIAL_HEAD_DIM],
  ).values;
  const decodeSin = readOfficialBf16Tensor(
    root,
    M6E7_OFFICIAL_PATHS.decodeFixture,
    "decoder.decode.00.rope.sin.axis_major",
    [3, 1, M6E7_OFFICIAL_HEAD_DIM],
  ).values;

  const ropeElements = 3 * M6E7_OFFICIAL_CAPACITY * M6E7_OFFICIAL_HEAD_DIM;
  const ropeCos = new Float32Array(ropeElements);
  const ropeSin = new Float32Array(ropeElements);
  for (let axis = 0; axis < 3; axis += 1) {
    const prefillBegin = axis * M6E7_OFFICIAL_TOKENS * M6E7_OFFICIAL_HEAD_DIM;
    const prefillEnd = (axis + 1) * M6E7_OFFICIAL_TOKENS * M6E7_OFFICIAL_HEAD_DIM;
    const outputBegin = axis * M6E7_OFFICIAL_CAPACITY * M6E7_OFFICIAL_HEAD_DIM;
    ropeCos.set(prefillCos.subarray(prefillBegin, prefillEnd), outputBegin);
    ropeSin.set(prefillSin.subarray(prefillBegin, prefillEnd), outputBegin);
    const decodeBegin = axis * M6E7_OFFICIAL_HEAD_DIM;
    const decodeEnd = (axis + 1) * M6E7_OFFICIAL_HEAD_DIM;
    const decodeOutput =
      (axis * M6E7_OFFICIAL_CAPACITY + M6E7_OFFICIAL_TOKENS) *
      M6E7_OFFICIAL_HEAD_DIM;
    ropeCos.set(decodeCos.subarray(decodeBegin, decodeEnd), decodeOutput);
    ropeSin.set(decodeSin.subarray(decodeBegin, decodeEnd), decodeOutput);
  }
  return {
    "weights.mrope_cos": new Uint8Array(ropeCos.buffer),
    "weights.mrope_sin": new Uint8Array(ropeSin.buffer),
  };
}

export async function assembleM7q1OfficialFp16Pack({
  modelPath,
  descriptor,
  compilerBuild,
  ropeTables,
  precisionProfile = "balanced",
  identifyCheckpoint = checkpointIdentity,
  readF16Tensor = readOfficialF16Tensor,
  buildBalancedWeightPack = buildM7q1BalancedWeightPack,
}) {
  invariant(typeof modelPath === "string", "modelPath is missing");
  invariant(typeof identifyCheckpoint === "function", "identity authority is missing");
  invariant(typeof readF16Tensor === "function", "F16 tensor reader is missing");
  invariant(
    typeof buildBalancedWeightPack === "function",
    "balanced pack builder is missing",
  );
  invariant(
    ropeTables !== null && typeof ropeTables === "object",
    "M-RoPE tables are missing",
  );

  // Fix the whole-file identity before opening any tensor payload. Every
  // subsequent positioned read uses this one exact path.
  const identity = await identifyCheckpoint(modelPath);
  const checkpointShards = new Map();
  for (const [shardId, suffix, shape] of LAYER_WEIGHT_ROLES) {
    const layerParts = [];
    for (let layer = 0; layer < LAYERS; layer += 1) {
      layerParts.push(readF16Tensor(
        modelPath,
        `model.layers.${layer}.${suffix}`,
        shape,
      ));
    }
    checkpointShards.set(shardId, concatenate(layerParts, shardId));
  }

  checkpointShards.set(
    "weights.final_layernorm",
    readF16Tensor(modelPath, "model.norm.weight", [1024]),
  );
  checkpointShards.set(
    "weights.lm_head",
    readF16Tensor(modelPath, "lm_head.weight", [103424, 1024]),
  );

  const shards = {};
  for (const id of M7Q1_LOGITS_WEIGHT_SHARD_IDS) {
    const value = id === "weights.mrope_cos" || id === "weights.mrope_sin"
      ? ropeTables[id]
      : checkpointShards.get(id);
    invariant(value instanceof Uint8Array, `${id} is missing`);
    shards[id] = value;
  }

  return buildBalancedWeightPack({
    layout: "logits",
    checkpointIdentity: identity,
    compilerBuild,
    descriptor,
    shards,
    precisionProfile,
  });
}
