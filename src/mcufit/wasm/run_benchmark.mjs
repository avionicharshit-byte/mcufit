// node run_benchmark.mjs <model.tflite>
// Runs the wasm32 TFLM benchmark and replays its output. Same tflm.js the
// website loads; emscripten built it with -sENVIRONMENT=web,node.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const modelPath = process.argv[2];
if (!modelPath) {
  console.error("usage: run_benchmark.mjs <model.tflite>");
  process.exit(2);
}

const factory = require(join(here, "tflm.js"));

const lines = [];
const mod = await factory({
  print: (t) => lines.push(t),
  printErr: (t) => lines.push(t),
});

mod.FS.writeFile("/model.tflite", readFileSync(modelPath));
try {
  mod.callMain(["/model.tflite"]);
} catch {
  // emscripten throws ExitStatus even on a clean exit; output is already ours
}

process.stdout.write(lines.join("\n") + "\n");
