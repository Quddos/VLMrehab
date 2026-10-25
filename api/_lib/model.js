// Loads each RehabVLM model once per function instance and reuses it across requests.
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { HAND_QUESTIONS, QUESTIONS, loadModel } from "../../web/core.mjs";

const DIRS = { body: join(process.cwd(), "web", "model"), hand: join(process.cwd(), "web", "model", "hand") };
const loading = {};

export function getModel(domain = "body") {
  if (!loading[domain]) {
    ort.env.wasm.numThreads = 1;
    const dir = DIRS[domain];
    loading[domain] = (async () => {
      const vocab = JSON.parse(await readFile(join(dir, "vocab.json"), "utf8"));
      return loadModel(ort, vocab, (f) => readFile(join(dir, f)), domain === "hand" ? HAND_QUESTIONS : QUESTIONS);
    })();
    loading[domain].catch(() => { delete loading[domain]; });
  }
  return loading[domain];
}
