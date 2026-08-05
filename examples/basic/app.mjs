import {
  createPaddleOcrVlEngine,
} from "../../web/engine/browser_ocr_runtime.mjs";
import { diagnoseBrowser } from "../../web/browser_diagnostics.mjs";

const form = document.querySelector("#recognition-form");
const imageInput = document.querySelector("#image-input");
const taskInput = document.querySelector("#task-input");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const submit = form.querySelector("button");

let enginePromise;

function engine() {
  enginePromise ??= createPaddleOcrVlEngine();
  return enginePromise;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files?.[0];
  if (!file) return;

  submit.disabled = true;
  result.textContent = "";
  try {
    const diagnostics = await diagnoseBrowser();
    if (!diagnostics.ready) {
      throw new Error(diagnostics.problems.join(" "));
    }
    status.textContent = "Preparing WebGPU runtime…";
    const output = await (await engine()).recognizeImage(file, {
      task: taskInput.value,
      onProgress(update) {
        if (update.kind === "download") {
          const percent = update.modelTotalBytes
            ? Math.round(update.modelLoadedBytes / update.modelTotalBytes * 100)
            : 0;
          status.textContent = `Downloading model assets… ${percent}%`;
        } else if (update.stage) {
          status.textContent = `${update.stage}: ${update.detail ?? ""}`;
        }
      },
    });
    status.textContent =
      `${output.greedy.tokenIds.length} tokens · ` +
      `${output.tokensPerSecond.toFixed(1)} tokens/s`;
    result.textContent = output.text;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    submit.disabled = false;
  }
});

window.addEventListener("pagehide", () => {
  enginePromise?.then((instance) => instance.dispose());
});
