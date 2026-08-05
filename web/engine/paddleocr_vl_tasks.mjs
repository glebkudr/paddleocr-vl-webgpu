export const PADDLEOCR_VL_TASKS = Object.freeze({
  ocr: Object.freeze({
    prompt: "OCR:",
    promptSuffixIds: Object.freeze([
      101306, 93972, 2497, 93963, 23, 92267, 93963, 23,
    ]),
  }),
  table: Object.freeze({
    prompt: "Table Recognition:",
    promptSuffixIds: Object.freeze([
      101306, 2567, 93514, 93963, 23, 92267, 93963, 23,
    ]),
  }),
  formula: Object.freeze({
    prompt: "Formula Recognition:",
    promptSuffixIds: Object.freeze([
      101306, 59352, 93514, 93963, 23, 92267, 93963, 23,
    ]),
  }),
  chart: Object.freeze({
    prompt: "Chart Recognition:",
    promptSuffixIds: Object.freeze([
      101306, 17720, 93514, 93963, 23, 92267, 93963, 23,
    ]),
  }),
});

export function recognitionOptionsForTask(task, {
  maxGeneratedTokens = 512,
} = {}) {
  const definition = PADDLEOCR_VL_TASKS[task];
  if (!definition) {
    throw new RangeError(
      `Unsupported PaddleOCR-VL task: ${String(task)}`,
    );
  }
  return Object.freeze({
    task,
    ...definition,
    maxGeneratedTokens:
      task === "table"
        ? Math.max(1024, maxGeneratedTokens)
        : maxGeneratedTokens,
  });
}
