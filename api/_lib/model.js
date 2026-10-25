// Loads RehabVLM once per function instance and reuses it across requests.
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import * as ort from "onnxruntime-web";
import { loadModel } from "../../web/core.mjs";

const DIR = join(process.cwd(), "web", "model");
let loading;

export function getModel() {
  if (!loading) {
    ort.env.wasm.numThreads = 1;
    loading = (async () => {
      const vocab = JSON.parse(await readFile(join(DIR, "vocab.json"), "utf8"));
      return loadModel(ort, vocab, (f) => readFile(join(DIR, f)));
    })();
    loading.catch(() => { loading = undefined; });
  }
  return loading;
}
