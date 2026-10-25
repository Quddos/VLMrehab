// RehabVLM core, shared by the browser app (web/app.js) and the HTTP API (api/).
// normalise(), rasterise() and measure() mirror rehabvlm/pose.py exactly; keep them in sync.

export const IMG = 96;
export const MODEL_VERSION = "rehabvlm-body-0.1";
export const HAND_MODEL_VERSION = "rehabvlm-hand-0.1";

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

export const HAND_QUESTIONS = {
  describe: ["Describe hand", "describe the hand ."],
  fist: ["Fist", "assess the fist ."],
  pinch: ["Pinch", "assess the pinch ."],
  limited: ["Most limited finger?", "which finger is most limited ?"],
  spread: ["Finger spread", "check finger spread ."],
  exercise: ["Suggest exercises", "suggest hand exercises ."],
};
const HAND_ROUTES = [
  [/exercis|rehab|therap|improve|help|train|workout|stretch/, "exercise"],
  [/pinch|grasp small|thumb.*index|index.*thumb|pick/, "pinch"],
  [/spread|apart|fan|abduct/, "spread"],
  [/limit|stiff|weak|worst|which finger|least/, "limited"],
  [/fist|grip|close|squeeze|curl|grab/, "fist"],
];

/** Map a free-text question (or a task name) to one of the supported tasks. */
export function routeQuestion(q, domain = "body") {
  const [questions, routes] = domain === "hand" ? [HAND_QUESTIONS, HAND_ROUTES] : [QUESTIONS, ROUTES];
  q = String(q || "").toLowerCase().trim();
  if (q in questions) return q;
  return (routes.find(([re]) => re.test(q)) || [null, "describe"])[1];
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

// Max-blend an anti-aliased segment p-q of the given width into channel ch.
function inkSegment(img, ch, p, q, width, size) {
  const dx = q[0] - p[0], dy = q[1] - p[1];
  const L2 = dx * dx + dy * dy + 1e-9;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const px = x + 0.5, py = y + 0.5;
    const t = Math.min(1, Math.max(0, ((px - p[0]) * dx + (py - p[1]) * dy) / L2));
    const v = Math.min(1, Math.max(0, width / 2 + 0.5 - Math.hypot(px - (p[0] + t * dx), py - (p[1] + t * dy))));
    const i = ch * size * size + y * size + x;
    if (v > img[i]) img[i] = v;
  }
}

export function rasterise(pts, width = 2.2, size = IMG) {
  const img = new Float32Array(3 * size * size);
  for (const [a, b, ch] of SEGMENTS) inkSegment(img, ch, pts[IDX[a]], pts[IDX[b]], width, size);
  const n = pts[IDX.nose], r = 0.06 * size;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const v = Math.min(1, Math.max(0, width / 2 + 0.5 - Math.abs(Math.hypot(x + 0.5 - n[0], y + 0.5 - n[1]) - r)));
    const i = 2 * size * size + y * size + x;
    if (v > img[i]) img[i] = v;
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
export async function loadModel(ort, vocabJson, load, questions = QUESTIONS) {
  const vocab = vocabJson.vocab;
  const [encoder, decoder] = await Promise.all([
    ort.InferenceSession.create(await load("encoder.onnx")),
    ort.InferenceSession.create(await load("decoder.onnx")),
  ]);
  return { ort, vocab, questions, stoi: Object.fromEntries(vocab.map((w, i) => [w, i])), encoder, decoder };
}

export async function encodePose(model, kps) {
  return encodePixels(model, rasterise(normalise(kps)));
}

export async function encodeHand(model, landmarks) {
  return encodePixels(model, rasteriseHand(canonicaliseHand(landmarks)));
}

async function encodePixels(model, pixels) {
  const { img_tokens } = await model.encoder.run({
    image: new model.ort.Tensor("float32", pixels, [1, 3, IMG, IMG]),
  });
  return { pixels, mem: img_tokens };
}

/** Greedy decoding. Returns the answer, or null if shouldStop() turned true. */
export async function generate(model, mem, task, { onToken, shouldStop } = {}) {
  const { ort, stoi, decoder } = model;
  const prompt = [BOS, ...model.questions[task][1].split(" ").map((w) => stoi[w]), SEP];
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

// ---------- hands (mirrors rehabvlm/hand.py) ----------
// 21 landmarks, MediaPipe Hands order: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20
export const HAND_FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
const HAND_CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
export const HAND_SEGMENTS = [
  ...HAND_CHAINS.flatMap((c) => [0, 1, 2, 3].map((i) => [c[i], c[i + 1]])), [5, 9], [9, 13], [13, 17],
];
const HAND_VIEWS = [[0, 1, 0.0, 1.0], [2, 1, 0.3, 1.0], [0, 2, 0.0, 0.3]];   // front, side, top
const HAND_PX = 36;

const v3 = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  norm: (a) => Math.hypot(a[0], a[1], a[2]),
};
const unit = (a) => v3.scale(a, 1 / (v3.norm(a) + 1e-9));
const ang3 = (a, b) =>
  Math.acos(Math.max(-1, Math.min(1, v3.dot(a, b) / (v3.norm(a) * v3.norm(b) + 1e-9)))) * 180 / Math.PI;

export function canonicaliseHand(k) {
  k = k.map((p) => v3.sub(p, k[0]));
  const y = unit(k[9]);
  const xt = v3.sub(k[5], k[17]);
  const x = unit(v3.sub(xt, v3.scale(y, v3.dot(xt, y))));
  const z = v3.cross(x, y);
  const s = 1 / (v3.norm(k[9]) + 1e-9);
  return k.map((p) => [v3.dot(p, x) * s, v3.dot(p, y) * s, v3.dot(p, z) * s]);
}

export function rasteriseHand(c, width = 2.0, size = IMG) {
  const img = new Float32Array(3 * size * size);
  HAND_VIEWS.forEach(([u, v, cu, cv], ch) => {
    const pts = c.map((p) => [(p[u] - cu) * HAND_PX + size / 2, size / 2 - (p[v] - cv) * HAND_PX]);
    for (const [a, b] of HAND_SEGMENTS) inkSegment(img, ch, pts[a], pts[b], width, size);
    for (const tip of [4, 8, 12, 16, 20]) inkSegment(img, ch, pts[tip], pts[tip], 2 * width, size);
  });
  return img;
}

/** Rotation-invariant finger measurements (same keys as rehabvlm/hand.py). */
export function measureHand(k) {
  const normal = unit(v3.cross(v3.sub(k[5], k[0]), v3.sub(k[17], k[0])));
  const out = {};
  HAND_FINGERS.forEach((name, f) => {
    const c = HAND_CHAINS[f];
    const segs = [0, 1, 2, 3].map((i) => v3.sub(k[c[i + 1]], k[c[i]]));
    let total = ang3(segs[1], segs[2]) + ang3(segs[2], segs[3]);
    if (name !== "thumb") {
      total += Math.asin(Math.min(1, Math.abs(v3.dot(unit(segs[1]), normal)))) * 180 / Math.PI;
    }
    out[`${name}_flex`] = total;
  });
  const proj = (v) => v3.sub(v, v3.scale(normal, v3.dot(v, normal)));
  const a = v3.sub(k[6], k[5]), b = v3.sub(k[18], k[17]);
  const flat = Math.min(v3.norm(proj(a)) / v3.norm(a), v3.norm(proj(b)) / v3.norm(b));
  out.spread = flat > 0.7 ? ang3(proj(a), proj(b)) : null;   // not measurable while fingers are curled
  out.pinch_gap = 100 * v3.norm(v3.sub(k[4], k[8])) / (v3.norm(v3.sub(k[5], k[17])) + 1e-9);
  return out;
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

/**
 * Hand input: { hand: [[x, y, z], ... 21 landmarks in MediaPipe order], space?: "image" | "unity" }.
 * Any orientation, scale or units work; only the hand's shape matters. Unity (left-handed axes) is
 * converted by flipping z.
 */
export function handFromInput(body) {
  const h = body.hand;
  if (!Array.isArray(h) || h.length !== 21) throw new Error("`hand` must be 21 landmarks [x, y, z]");
  return h.map((p) => {
    if (!Array.isArray(p) || p.length < 3 || !p.slice(0, 3).every(Number.isFinite)) {
      throw new Error("each hand landmark needs [x, y, z], got " + JSON.stringify(p));
    }
    return body.space === "unity" ? [p[0], p[1], -p[2]] : [p[0], p[1], p[2]];
  });
}
