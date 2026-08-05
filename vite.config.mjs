import { defineConfig } from "vite";
import { copyFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: "examples/basic",
  base: "./",
  plugins: [{
    name: "copy-public-legal-notices",
    async closeBundle() {
      await Promise.all([
        ["LICENSE", "LICENSE.txt"],
        ["NOTICE", "NOTICE.txt"],
        ["THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"],
      ].map(([source, destination]) =>
        copyFile(
          resolve(repositoryRoot, source),
          resolve(repositoryRoot, "dist", destination),
        )
      ));
    },
  }],
  build: {
    outDir: "../../dist",
    emptyOutDir: true,
    target: "es2022",
  },
});
