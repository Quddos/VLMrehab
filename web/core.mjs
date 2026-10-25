// RehabVLM core, shared by the browser app (web/app.js) and the HTTP API (api/).
// normalise(), rasterise() and measure() mirror rehabvlm/pose.py exactly; keep them in sync.

export const IMG = 96;
export const MODEL_VERSION = "rehabvlm-body-0.1";

export const QUESTIONS = {
  describe: ["Describe posture", "describe the posture ."],
  shoulders: ["Shoulders", "assess the shoulders ."],
  elbows: ["Elbows", "assess the elbows ."],
  hips: ["Hips", "assess the hips ."],
  symmetry: ["Symmetry", "check symmetry ."],
  weaker: ["Weaker side?", "which side is weaker ?"],
  exercise: ["Suggest exercises", "suggest exercises ."],
};
const ROUTES = [
  [/exercis|rehab|therap|improve|help|train|workout|stretch/, "exercise"],
  [/weak|affected|worse|stronger|which side/, "weaker"],
  [/symmetr|balanc|compar|even|both sides/, "symmetry"],
  [/elbow|bend|forearm/, "elbows"],
  [/hip|leg|stance|feet|knee/, "hips"],
  [/shoulder|arm|raise|lift|reach|abduct/, "shoulders"],
];

/** Map a free-text question (or a task name) to one of the supported tasks. */
export function routeQuestion(q) {
  q = String(q || "").toLowerCase().trim();
  if (q in QUESTIONS) return q;
  return (ROUTES.find(([re]) => re.test(q)) || [null, "describe"])[1];
}

export const KP = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
  "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"];
export const MP_INDEX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const NAMES = [...KP, "neck", "midhip"];
const IDX = Object.fromEntries(NAMES.map((n, i) => [n, i]));
export const SEGMENTS = [
  ["r_shoulder", "r_elbow", 0], ["r_elbow", "r_wrist", 0],
  ["r_hip", "r_knee", 0], ["r_knee", "r_ankle", 0],
  ["l_shoulder", "l_elbow", 1], ["l_elbow", "l_wrist", 1],
  ["l_hip", "l_knee", 1], ["l_knee", "l_ankle", 1],
  ["l_shoulder", "r_shoulder", 2], ["l_hip", "r_hip", 2],
  ["neck", "midhip", 2], ["neck", "nose", 2],
];
export const SEGMENT_INDEX = IDX;
const BOS = 1, SEP = 2, EOS = 3;

// ---------- geometry ----------
export const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];

export function normalise(k, size = IMG, margin = 0.1) {
  const pts = [...k, mid(k[1], k[2]), mid(k[7], k[8])];
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of pts) for (let d = 0; d < 2; d++) {
    lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]);
  }
  const scale = (1 - 2 * margin) * size / Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-6);
  const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
  return pts.map((p) => [(p[0] - c[0]) * scale + size / 2, (p[1] - c[1]) * scale + size / 2]);
}

export function rasterise(pts, width = 2.2, size = IMG) {
  const img = new Float32Array(3 * size * size);
  const ink = (ch, dist, x, y) => {
    const v = Math.min(1, Math.max(0, width / 2 + 0.5 - dist));
    const i = ch * size * size + y * size + x;
    if (v > img[i]) img[i] = v;
  };
  for (const [a, b, ch] of SEGMENTS) {
    const p = pts[IDX[a]], q = pts[IDX[b]];
    const dx = q[0] - p[0], dy = q[1] - p[1];
    const L2 = dx * dx + dy * dy + 1e-9;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const px = x + 0.5, py = y + 0.5;
      const t = Math.min(1, Math.max(0, ((px - p[0]) * dx + (py - p[1]) * dy) / L2));
      ink(ch, Math.hypot(px - (p[0] + t * dx), py - (p[1] + t * dy)), x, y);
    }
  }
  const n = pts[IDX.nose], r = 0.06 * size;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    ink(2, Math.abs(Math.hypot(x + 0.5 - n[0], y + 0.5 - n[1]) - r), x, y);
  }
  return img;
}

function angle(a, b) {
  const c = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b) + 1e-9);
  return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
}

/** Joint angles in degrees measured directly from keypoints (same keys as rehabvlm/pose.py). */
export function measure(k) {
  const g = (n) => k[KP.indexOf(n)];
  const neck = mid(g("l_shoulder"), g("r_shoulder")), midhip = mid(g("l_hip"), g("r_hip"));
  const down = sub(midhip, neck), out = {};
  for (const s of ["r", "l"]) {
    out[`${s}_shoulder_abd`] = angle(sub(g(`${s}_elbow`), g(`${s}_shoulder`)), down);
    out[`${s}_elbow_flex`] =
      180 - angle(sub(g(`${s}_shoulder`), g(`${s}_elbow`)), sub(g(`${s}_wrist`), g(`${s}_elbow`)));
    out[`${s}_hip_abd`] = angle(sub(g(`${s}_knee`), g(`${s}_hip`)), down);
  }
  const up = sub(neck, midhip);
  out.trunk_lean = Math.atan2(up[0], -up[1]) * 180 / Math.PI;   // + = towards subject's left
  return out;
}

// ---------- model ----------
/** Load both ONNX graphs. `load(path)` returns what ort.InferenceSession.create accepts. */
export async function loadModel(ort, vocabJson, load) {
  const vocab = vocabJson.vocab;
  const [encoder, decoder] = await Promise.all([
    ort.InferenceSession.create(await load("encoder.onnx")),
    ort.InferenceSession.create(await load("decoder.onnx")),
  ]);
  return { ort, vocab, stoi: Object.fromEntries(vocab.map((w, i) => [w, i])), encoder, decoder };
}

export async function encodePose(model, kps) {
  const pixels = rasterise(normalise(kps));
  const { img_tokens } = await model.encoder.run({
    image: new model.ort.Tensor("float32", pixels, [1, 3, IMG, IMG]),
  });
  return { pixels, mem: img_tokens };
}

/** Greedy decoding. Returns the answer, or null if shouldStop() turned true. */
export async function generate(model, mem, task, { onToken, shouldStop } = {}) {
  const { ort, stoi, decoder } = model;
  const prompt = [BOS, ...QUESTIONS[task][1].split(" ").map((w) => stoi[w]), SEP];
  const ids = [...prompt];
  for (let step = 0; step < 64; step++) {
    if (shouldStop && shouldStop()) return null;
    const { logits } = await decoder.run({
      img_tokens: mem,
      ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    });
    const V = logits.dims[2], off = (logits.dims[1] - 1) * V;
    let best = 0;
    for (let i = 1; i < V; i++) if (logits.data[off + i] > logits.data[off + best]) best = i;
    if (best === EOS) break;
    ids.push(best);
    if (onToken) onToken(pretty(model.vocab, ids.slice(prompt.length)));
  }
  return pretty(model.vocab, ids.slice(prompt.length));
}

export function pretty(vocab, ids) {
  const s = ids.filter((i) => i > 4).map((i) => vocab[i]).join(" ")
    .replace(/ \./g, ".").replace(/ ,/g, ",").replace(/ \?/g, "?");
  return s.replace(/(^|\. )([a-z])/g, (m, p, c) => p + c.toUpperCase());
}

// ---------- input adapters (used by the API) ----------
/**
 * Convert API input into 13 image-space keypoints [[x, y], ...] (x right, y down,
 * subject facing the viewer so their LEFT side is at larger x).
 *
 * Accepted shapes:
 *  - { joints: { nose: [x,y(,z)], l_shoulder: ..., ... }, space: "image" | "unity" }
 *  - { mediapipe: [[x,y(,z)], ... 33 landmarks] }
 * space "unity": positions in the avatar's root space (Y up, avatar faces +Z), as sent by
 * the Unity client. They are projected onto the frontal plane as seen from the front.
 */
export function keypointsFromInput(body) {
  let kps;
  if (Array.isArray(body.mediapipe)) {
    if (body.mediapipe.length < 29) throw new Error("mediapipe needs the 33 pose landmarks");
    kps = MP_INDEX.map((m) => body.mediapipe[m]);
  } else if (body.joints && typeof body.joints === "object") {
    kps = KP.map((n) => body.joints[n]);
    const missing = KP.filter((n, i) => !Array.isArray(kps[i]));
    if (missing.length) throw new Error("missing joints: " + missing.join(", "));
  } else {
    throw new Error("send either `joints` (named keypoints) or `mediapipe` (33 landmarks)");
  }
  kps = kps.map((p) => {
    if (!Array.isArray(p) || p.length < 2 || !p.every(Number.isFinite)) throw new Error("bad point " + JSON.stringify(p));
    return body.space === "unity" ? [-p[0], -p[1]] : [p[0], p[1]];
  });
  return kps;
}
