const HIDDEN = 1152;
const INTERMEDIATE = 4304;

export const VISION_LAYER_TENSOR_ROLES = Object.freeze([
  Object.freeze(["layer_norm1.weight", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze(["layer_norm1.bias", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze([
    "self_attn.q_proj.weight",
    Object.freeze([HIDDEN, HIDDEN]),
    "f16",
  ]),
  Object.freeze(["self_attn.q_proj.bias", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze([
    "self_attn.k_proj.weight",
    Object.freeze([HIDDEN, HIDDEN]),
    "f16",
  ]),
  Object.freeze(["self_attn.k_proj.bias", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze([
    "self_attn.v_proj.weight",
    Object.freeze([HIDDEN, HIDDEN]),
    "f16",
  ]),
  Object.freeze(["self_attn.v_proj.bias", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze([
    "self_attn.out_proj.weight",
    Object.freeze([HIDDEN, HIDDEN]),
    "f16",
  ]),
  Object.freeze(["self_attn.out_proj.bias", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze(["layer_norm2.weight", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze(["layer_norm2.bias", Object.freeze([HIDDEN]), "f32"]),
  Object.freeze([
    "mlp.fc1.weight",
    Object.freeze([INTERMEDIATE, HIDDEN]),
    "f16",
  ]),
  Object.freeze(["mlp.fc1.bias", Object.freeze([INTERMEDIATE]), "f32"]),
  Object.freeze([
    "mlp.fc2.weight",
    Object.freeze([HIDDEN, INTERMEDIATE]),
    "f16",
  ]),
  Object.freeze(["mlp.fc2.bias", Object.freeze([HIDDEN]), "f32"]),
]);

function invariant(condition, message) {
  if (!condition) {
    throw new Error(`FP16 matrix layout: ${message}`);
  }
}

function positiveSafeDimension(value, label) {
  invariant(
    Number.isSafeInteger(value) && value > 0,
    `${label} must be a positive safe integer`,
  );
}

export function transposeF16OutputMajorToInputMajor(
  source,
  outputs,
  inputs,
  label,
) {
  invariant(source instanceof Uint8Array, `${label} must be a Uint8Array`);
  positiveSafeDimension(outputs, `${label} output count`);
  positiveSafeDimension(inputs, `${label} input count`);
  const elements = outputs * inputs;
  invariant(Number.isSafeInteger(elements), `${label} element count overflows`);
  const expectedBytes = elements * 2;
  invariant(Number.isSafeInteger(expectedBytes), `${label} byte length overflows`);
  invariant(
    source.byteLength === expectedBytes,
    `${label} byte length ${source.byteLength} does not match ${expectedBytes}`,
  );

  const input = new DataView(source.buffer, source.byteOffset, source.byteLength);
  const output = new Uint8Array(expectedBytes);
  const transposed = new DataView(output.buffer);
  for (let outputIndex = 0; outputIndex < outputs; outputIndex += 1) {
    for (let inputIndex = 0; inputIndex < inputs; inputIndex += 1) {
      const sourceIndex = outputIndex * inputs + inputIndex;
      const destinationIndex = inputIndex * outputs + outputIndex;
      transposed.setUint16(
        destinationIndex * 2,
        input.getUint16(sourceIndex * 2, true),
        true,
      );
    }
  }
  return output;
}

export function materializeVisionTensor({
  raw,
  shape,
  storage,
  label,
  transpose = transposeF16OutputMajorToInputMajor,
  widen,
}) {
  invariant(raw instanceof Uint8Array, `${label} must be a Uint8Array`);
  invariant(Array.isArray(shape) || Object.isFrozen(shape), `${label} shape is invalid`);
  invariant(typeof label === "string" && label.length > 0, "tensor label is invalid");
  if (storage === "f16") {
    if (shape.length === 1) {
      invariant(
        raw.byteLength === shape[0] * 2,
        `${label} F16 vector byte length drifted`,
      );
      return raw.slice();
    }
    invariant(shape.length === 2, `${label} F16 tensor must have rank one or two`);
    invariant(typeof transpose === "function", `${label} transpose callback is invalid`);
    return transpose(raw, shape[0], shape[1], label);
  }
  invariant(storage === "f32", `${label} storage is invalid`);
  invariant(shape.length === 1, `${label} F32 vector must be rank one`);
  invariant(typeof widen === "function", `${label} widen callback is invalid`);
  return widen(raw, label);
}
