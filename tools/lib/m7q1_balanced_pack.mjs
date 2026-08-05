import { m6e6FastBlake3Hex } from "../../web/tests/support/m6e6_decoder_stack_session_oracle.mjs";

const MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6";
const MODEL_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const DESCRIPTOR_ID = "ir.decoder_stack_00";
const HEADER_BYTES = 32;
const DIRECTORY_FIXED_BYTES = 56;
const SECTION_ALIGNMENT = 256;
const TEXT_ENCODER = new TextEncoder();

export const M7Q1_CHECKPOINT_WEIGHT_SHARD_IDS = Object.freeze([
  "weights.input_layernorm",
  "weights.q_proj",
  "weights.k_proj",
  "weights.v_proj",
  "weights.o_proj",
  "weights.post_attention_layernorm",
  "weights.gate_proj",
  "weights.up_proj",
  "weights.down_proj",
]);

export const M7Q1_ROPE_TABLE_SHARD_IDS = Object.freeze([
  "weights.mrope_cos",
  "weights.mrope_sin",
]);

export const M7Q1_LEGACY_WEIGHT_SHARD_IDS = Object.freeze([
  ...M7Q1_CHECKPOINT_WEIGHT_SHARD_IDS.slice(0, 5),
  ...M7Q1_ROPE_TABLE_SHARD_IDS,
  ...M7Q1_CHECKPOINT_WEIGHT_SHARD_IDS.slice(5),
]);

export const M7Q1_LOGITS_WEIGHT_SHARD_IDS = Object.freeze([
  ...M7Q1_LEGACY_WEIGHT_SHARD_IDS,
  "weights.final_layernorm",
  "weights.lm_head",
]);

function fail(message) {
  throw new Error(`M7q1 balanced pack: ${message}`);
}

function invariant(condition, message) {
  if (!condition) fail(message);
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function alignUp(value, alignment) {
  const remainder = value % alignment;
  return remainder === 0 ? value : value + alignment - remainder;
}

function canonicalJson(value) {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    invariant(Number.isFinite(value), "canonical JSON contains a nonfinite number");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  invariant(isRecord(value), "canonical JSON contains an unsupported value");
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(",")}}`;
}

function canonicalJsonBytes(value) {
  return TEXT_ENCODER.encode(`${canonicalJson(value)}\n`);
}

function digestBytes(hex) {
  invariant(/^[0-9a-f]{64}$/.test(hex), "section digest is not BLAKE3 hex");
  const bytes = new Uint8Array(32);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

function requireOwnedBytes(value, id) {
  invariant(value instanceof Uint8Array, `${id} is missing or not Uint8Array`);
  invariant(value.byteLength > 0, `${id} is empty`);
  return value;
}

function requireFiniteF16(bytes, id) {
  invariant(bytes.byteLength % 2 === 0, `${id} is not F16 aligned`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < bytes.byteLength / 2; index += 1) {
    const bits = view.getUint16(index * 2, true);
    invariant(
      (bits & 0x7c00) !== 0x7c00,
      `${id} contains nonfinite F16 at ${index}`,
    );
  }
}

function requireFiniteF32(bytes, id) {
  invariant(bytes.byteLength % 4 === 0, `${id} is not F32 aligned`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < bytes.byteLength / 4; index += 1) {
    invariant(
      Number.isFinite(view.getFloat32(index * 4, true)),
      `${id} contains nonfinite F32 at ${index}`,
    );
  }
}

function requireDescriptor(descriptor) {
  invariant(isRecord(descriptor), "descriptor is missing");
  for (const [field, expected] of [
    ["schema_version", 1],
    ["model_revision", MODEL_REVISION],
    ["layers", 18],
    ["hidden_size", 1024],
    ["intermediate_size", 3072],
    ["query_heads", 16],
    ["key_value_heads", 2],
    ["head_dim", 128],
    ["query_width", 2048],
    ["key_value_width", 256],
  ]) {
    invariant(descriptor[field] === expected, `descriptor ${field} drifted`);
  }
  invariant(
    descriptor.oracle === "synthetic" || descriptor.oracle === "official_l3",
    "descriptor oracle is unsupported",
  );
  invariant(
    typeof descriptor.case_id === "string" && descriptor.case_id.length > 0,
    "descriptor case_id is invalid",
  );
  invariant(
    Number.isSafeInteger(descriptor.prefix_tokens) && descriptor.prefix_tokens >= 0,
    "descriptor prefix_tokens is invalid",
  );
  invariant(
    Number.isSafeInteger(descriptor.cache_capacity) &&
      descriptor.cache_capacity > descriptor.prefix_tokens,
    "descriptor cache_capacity is invalid",
  );
  invariant(
    descriptor.rms_norm_epsilon === 1e-5,
    "descriptor rms_norm_epsilon drifted",
  );
  invariant(
    JSON.stringify(descriptor.mrope_sections) === JSON.stringify([16, 24, 24]),
    "descriptor mrope_sections drifted",
  );
}

function descriptorWithPins(descriptor, checkpointIdentity, payloadEntries) {
  return {
    schema_version: descriptor.schema_version,
    oracle: descriptor.oracle,
    case_id: descriptor.case_id,
    model_revision: descriptor.model_revision,
    layers: descriptor.layers,
    hidden_size: descriptor.hidden_size,
    intermediate_size: descriptor.intermediate_size,
    query_heads: descriptor.query_heads,
    key_value_heads: descriptor.key_value_heads,
    head_dim: descriptor.head_dim,
    query_width: descriptor.query_width,
    key_value_width: descriptor.key_value_width,
    prefix_tokens: descriptor.prefix_tokens,
    cache_capacity: descriptor.cache_capacity,
    rms_norm_epsilon: descriptor.rms_norm_epsilon,
    mrope_sections: [...descriptor.mrope_sections],
    checkpoint_blake3: checkpointIdentity.checkpoint_blake3,
    checkpoint_bytes: checkpointIdentity.checkpoint_bytes,
    weight_storage: "f16",
    shards: Object.fromEntries(payloadEntries.map((entry) => [
      entry.id,
      {
        bytes: entry.payload.byteLength,
        blake3: entry.blake3,
        dtype: entry.dtype,
      },
    ])),
  };
}

export function buildM7q1BalancedWeightPack({
  layout,
  checkpointIdentity,
  compilerBuild,
  descriptor,
  shards,
  precisionProfile = "balanced",
}) {
  invariant(layout === "legacy" || layout === "logits", "layout is unsupported");
  invariant(precisionProfile === "balanced", "precision profile must be balanced");
  invariant(isRecord(checkpointIdentity), "checkpoint identity is missing");
  invariant(
    checkpointIdentity.dtype === "float16" &&
      typeof checkpointIdentity.checkpoint_path === "string" &&
      /^[0-9a-f]{64}$/.test(checkpointIdentity.checkpoint_blake3) &&
      Number.isSafeInteger(checkpointIdentity.checkpoint_bytes) &&
      checkpointIdentity.checkpoint_bytes > 0,
    "checkpoint identity is invalid",
  );
  invariant(
    typeof compilerBuild === "string" && /^[0-9a-f]{64}$/.test(compilerBuild),
    "compiler build is not BLAKE3 hex",
  );
  requireDescriptor(descriptor);
  invariant(isRecord(shards), "shards are missing");

  const shardIds = layout === "legacy"
    ? M7Q1_LEGACY_WEIGHT_SHARD_IDS
    : M7Q1_LOGITS_WEIGHT_SHARD_IDS;
  invariant(
    Object.keys(shards).length === shardIds.length &&
      shardIds.every((id) => Object.hasOwn(shards, id)),
    `${layout} shard set drifted`,
  );

  const payloadEntries = shardIds.map((id) => {
    const payload = requireOwnedBytes(shards[id], id);
    const table = M7Q1_ROPE_TABLE_SHARD_IDS.includes(id);
    if (table) {
      requireFiniteF32(payload, id);
    } else {
      requireFiniteF16(payload, id);
    }
    return {
      id,
      kind: 2,
      payload,
      dtype: table ? "f32" : "f16",
      blake3: m6e6FastBlake3Hex(payload),
    };
  });
  const descriptorObject = descriptorWithPins(
    descriptor,
    checkpointIdentity,
    payloadEntries,
  );
  const descriptorPayload = canonicalJsonBytes(descriptorObject);
  const manifestPayload = canonicalJsonBytes({
    compiler_build: compilerBuild,
    compiler_model_abi: 1,
    context_limit: 4096,
    model_id: MODEL_ID,
    model_revision: MODEL_REVISION,
    precision_profile: precisionProfile,
    resolution_buckets: [[672, 672]],
  });
  const sections = [
    {
      id: DESCRIPTOR_ID,
      kind: 1,
      payload: descriptorPayload,
      blake3: m6e6FastBlake3Hex(descriptorPayload),
    },
    ...payloadEntries,
  ];

  const directoryEntries = sections.map((section) => {
    const idBytes = TEXT_ENCODER.encode(section.id);
    return {
      ...section,
      idBytes,
      entryBytes: alignUp(DIRECTORY_FIXED_BYTES + idBytes.byteLength, 8),
    };
  });
  const directoryLength = directoryEntries.reduce(
    (total, entry) => total + entry.entryBytes,
    0,
  );
  let cursor = HEADER_BYTES + manifestPayload.byteLength + directoryLength;
  for (const entry of directoryEntries) {
    cursor = alignUp(cursor, SECTION_ALIGNMENT);
    entry.offset = cursor;
    cursor += entry.payload.byteLength;
  }

  const packBytes = new Uint8Array(cursor);
  const view = new DataView(packBytes.buffer);
  packBytes.set(TEXT_ENCODER.encode("PVLCPK01"), 0);
  view.setUint32(8, 1, true);
  view.setUint32(12, manifestPayload.byteLength, true);
  view.setUint32(16, directoryLength, true);
  view.setUint32(20, directoryEntries.length, true);
  view.setBigUint64(24, BigInt(packBytes.byteLength), true);
  packBytes.set(manifestPayload, HEADER_BYTES);

  let directoryCursor = HEADER_BYTES + manifestPayload.byteLength;
  for (const entry of directoryEntries) {
    view.setUint16(directoryCursor, entry.idBytes.byteLength, true);
    view.setUint8(directoryCursor + 2, entry.kind);
    view.setUint8(directoryCursor + 3, 0);
    view.setUint32(directoryCursor + 4, SECTION_ALIGNMENT, true);
    view.setBigUint64(directoryCursor + 8, BigInt(entry.offset), true);
    view.setBigUint64(
      directoryCursor + 16,
      BigInt(entry.payload.byteLength),
      true,
    );
    packBytes.set(digestBytes(entry.blake3), directoryCursor + 24);
    packBytes.set(entry.idBytes, directoryCursor + DIRECTORY_FIXED_BYTES);
    packBytes.set(entry.payload, entry.offset);
    directoryCursor += entry.entryBytes;
  }

  return {
    packBytes,
    descriptor: descriptorObject,
    sectionIndex: directoryEntries.map((entry) => ({
      id: entry.id,
      kind: entry.kind,
      offset: entry.offset,
      byteLength: entry.payload.byteLength,
      blake3: entry.blake3,
    })),
  };
}
