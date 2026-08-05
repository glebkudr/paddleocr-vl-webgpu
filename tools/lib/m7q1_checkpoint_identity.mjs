import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { realpath, stat } from "node:fs/promises";

const execFileAsync = promisify(execFile);
const MODULE_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const REPOSITORY_ROOT = resolve(MODULE_DIRECTORY, "..", "..");
const HASH_HELPER = resolve(MODULE_DIRECTORY, "m7q1_blake3_file.py");
const CHECKPOINT_FILE = "model.safetensors";

function pythonExecutable() {
  if (process.env.PVLC_PYTHON) return process.env.PVLC_PYTHON;
  const repositoryPython = resolve(REPOSITORY_ROOT, ".venv", "bin", "python");
  return existsSync(repositoryPython) ? repositoryPython : "python3";
}

function invariant(condition, message) {
  if (!condition) throw new Error(`M7q1 checkpoint identity: ${message}`);
}

export async function checkpointIdentity(checkpoint) {
  const requested = resolve(String(checkpoint));
  invariant(
    basename(requested) === CHECKPOINT_FILE,
    `checkpoint must be exactly ${CHECKPOINT_FILE}`,
  );

  const checkpointPath = await realpath(requested);
  const checkpointStat = await stat(checkpointPath);
  invariant(checkpointStat.isFile(), "checkpoint is not a regular file");

  let stdout;
  try {
    ({ stdout } = await execFileAsync(
      pythonExecutable(),
      [HASH_HELPER, checkpointPath],
      {
        cwd: REPOSITORY_ROOT,
        encoding: "utf8",
        maxBuffer: 1024 * 1024,
      },
    ));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`M7q1 checkpoint identity: BLAKE3 failed: ${detail}`);
  }

  let hashed;
  try {
    hashed = JSON.parse(stdout.trim());
  } catch {
    throw new Error("M7q1 checkpoint identity: BLAKE3 helper returned invalid JSON");
  }
  invariant(
    typeof hashed.blake3 === "string" && /^[0-9a-f]{64}$/.test(hashed.blake3),
    "BLAKE3 helper returned an invalid digest",
  );
  invariant(
    Number.isSafeInteger(hashed.bytes) && hashed.bytes === checkpointStat.size,
    "checkpoint byte length changed while hashing",
  );

  return {
    checkpoint_path: checkpointPath,
    checkpoint_blake3: hashed.blake3,
    checkpoint_bytes: hashed.bytes,
    dtype: "float16",
  };
}
