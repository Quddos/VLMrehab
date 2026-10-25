// RehabVLM browser runtime: pose -> skeleton raster -> tiny VLM (onnxruntime-web).
// normalise() and rasterise() mirror rehabvlm/pose.py exactly; keep them in sync.

const IMG = 96;
const MP_VERSION = "0.10.14";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const QUESTIONS = {
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

const KP = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
  "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"];
const MP_INDEX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const NAMES = [...KP, "neck", "midhip"];
const IDX = Object.fromEntries(NAMES.map((n, i) => [n, i]));
const SEGMENTS = [
  ["r_shoulder", "r_elbow", 0], ["r_elbow", "r_wrist", 0],
  ["r_hip", "r_knee", 0], ["r_knee", "r_ankle", 0],
  ["l_shoulder", "l_elbow", 1], ["l_elbow", "l_wrist", 1],
  ["l_hip", "l_knee", 1], ["l_knee", "l_ankle", 1],
  ["l_shoulder", "r_shoulder", 2], ["l_hip", "r_hip", 2],
  ["neck", "midhip", 2], ["neck", "nose", 2],
];
const PAD = 0, BOS = 1, SEP = 2, EOS = 3;

const $ = (id) => document.getElementById(id);
const status = (s) => { $("status").textContent = s; };

let vocab, stoi, encoder, decoder, landmarker;
let current = null;   // { kps: [[x,y]...13], pixels: Float32Array, mem: ort.Tensor }
let busy = false, pending = null, runId = 0;

// ---------- geometry (mirrors rehabvlm/pose.py) ----------
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

function normalise(k, size = IMG, margin = 0.1) {
  const pts = [...k, mid(k[1], k[2]), mid(k[7], k[8])];
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of pts) for (let d = 0; d < 2; d++) {
    lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]);
  }
  const scale = (1 - 2 * margin) * size / Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-6);
  const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
  return pts.map((p) => [(p[0] - c[0]) * scale + size / 2, (p[1] - c[1]) * scale + size / 2]);
}

function rasterise(pts, width = 2.2, size = IMG) {
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
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];

function measure(k) {
  const g = (n) => k[KP.indexOf(n)];
  const neck = mid(g("l_shoulder"), g("r_shoulder")), midhip = mid(g("l_hip"), g("r_hip"));
  const down = sub(midhip, neck), out = {};
  for (const [s, label] of [["r", "Right"], ["l", "Left"]]) {
    out[`${label} shoulder abduction`] = angle(sub(g(`${s}_elbow`), g(`${s}_shoulder`)), down);
    out[`${label} elbow flexion`] =
      180 - angle(sub(g(`${s}_shoulder`), g(`${s}_elbow`)), sub(g(`${s}_wrist`), g(`${s}_elbow`)));
    out[`${label} hip abduction`] = angle(sub(g(`${s}_knee`), g(`${s}_hip`)), down);
  }
  const up = sub(neck, midhip);
  out["Trunk lean (+ = to left)"] = Math.atan2(up[0], -up[1]) * 180 / Math.PI;
  return out;
}

// ---------- model ----------
async function loadModel() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const v = await (await fetch("model/vocab.json")).json();
  vocab = v.vocab;
  stoi = Object.fromEntries(vocab.map((w, i) => [w, i]));
  [encoder, decoder] = await Promise.all([
    ort.InferenceSession.create("model/encoder.onnx"),
    ort.InferenceSession.create("model/decoder.onnx"),
  ]);
}

function idsTensor(ids) {
  return new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
}

async function generate(task, onToken, id) {
  const prompt = [BOS, ...QUESTIONS[task][1].split(" ").map((w) => stoi[w]), SEP];
  const ids = [...prompt];
  for (let step = 0; step < 64; step++) {
    if (id !== runId) return null;   // superseded by a newer question or pose
    const { logits } = await decoder.run({ img_tokens: current.mem, ids: idsTensor(ids) });
    const V = logits.dims[2], T = logits.dims[1], off = (T - 1) * V;
    let best = 0;
    for (let i = 1; i < V; i++) if (logits.data[off + i] > logits.data[off + best]) best = i;
    if (best === EOS) break;
    ids.push(best);
    onToken(pretty(ids.slice(prompt.length)));
  }
  return pretty(ids.slice(prompt.length));
}

function pretty(ids) {
  const s = ids.filter((i) => i > 4).map((i) => vocab[i]).join(" ")
    .replace(/ \./g, ".").replace(/ ,/g, ",").replace(/ \?/g, "?");
  return s.replace(/(^|\. )([a-z])/g, (m, p, c) => p + c.toUpperCase());
}

// ---------- drawing ----------
function drawSkeleton(pixels) {
  const c = $("skeleton"), ctx = c.getContext("2d");
  const small = new ImageData(IMG, IMG);
  for (let i = 0; i < IMG * IMG; i++) {
    small.data[4 * i] = pixels[i] * 255;
    small.data[4 * i + 1] = pixels[IMG * IMG + i] * 255;
    small.data[4 * i + 2] = pixels[2 * IMG * IMG + i] * 255;
    small.data[4 * i + 3] = 255;
  }
  const tmp = new OffscreenCanvas(IMG, IMG);
  tmp.getContext("2d").putImageData(small, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, c.width, c.height);
}

function drawPhoto(source, kps, w, h) {
  const c = $("photo"), ctx = c.getContext("2d");
  const s = Math.min(c.width / w, c.height / h);
  const ox = (c.width - w * s) / 2, oy = (c.height - h * s) / 2;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
  if (source) ctx.drawImage(source, ox, oy, w * s, h * s);
  const P = (p) => [ox + p[0] * s, oy + p[1] * s];
  const colors = ["#ef4444", "#22c55e", "#3b82f6"];
  const all = [...kps, mid(kps[1], kps[2]), mid(kps[7], kps[8])];
  ctx.lineWidth = 3; ctx.lineCap = "round";
  for (const [a, b, ch] of SEGMENTS) {
    const p = P(all[IDX[a]]), q = P(all[IDX[b]]);
    ctx.strokeStyle = colors[ch]; ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  for (const p of kps) { const [x, y] = P(p); ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); }
}

function showGeometry(kps) {
  $("geom").innerHTML = Object.entries(measure(kps))
    .map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(0)}°</td></tr>`).join("");
}

// ---------- pipeline ----------
async function usePose(kps, source, w, h) {
  if ($("mirror").checked) {
    // a mirrored selfie swaps the subject's sides: flip x and swap left/right labels
    kps = kps.map((p) => [w - p[0], p[1]]);
    const swap = (i, j) => { [kps[i], kps[j]] = [kps[j], kps[i]]; };
    for (let i = 1; i < 13; i += 2) swap(i, i + 1);
  }
  const pixels = rasterise(normalise(kps));
  drawPhoto(source, kps, w, h);
  drawSkeleton(pixels);
  showGeometry(kps);
  const { img_tokens } = await encoder.run({ image: new ort.Tensor("float32", pixels, [1, 3, IMG, IMG]) });
  current = { kps, pixels, mem: img_tokens };
  status("Pose ready. Ask a question.");
  ask("describe");
}

async function ask(task) {
  if (!current) { status("Add a photo or pick an example first."); return; }
  const id = ++runId;
  if (busy) { pending = task; return; }
  busy = true;
  document.querySelectorAll("#questions .chip").forEach((b) => b.classList.toggle("on", b.dataset.task === task));
  const box = $("answer");
  box.innerHTML = `<span class="q">${QUESTIONS[task][0]}</span><span class="a"></span>`;
  const a = box.querySelector(".a");
  const t0 = performance.now();
  try {
    const done = await generate(task, (s) => { a.textContent = s; }, id);
    if (done !== null) status(`Answered in ${((performance.now() - t0) / 1000).toFixed(2)} s, on your device.`);
  } catch (e) {
    a.textContent = "Error: " + e.message;
  } finally {
    busy = false;
  }
  if (pending) { const next = pending; pending = null; ask(next); }
}

async function getLandmarker() {
  if (landmarker) return landmarker;
  status("Loading pose estimator (first time only)…");
  const base = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
  const { FilesetResolver, PoseLandmarker } = await import(`${base}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${base}/wasm`);
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL },
    runningMode: "IMAGE", numPoses: 1,
  });
  return landmarker;
}

async function onPhoto(file) {
  const bmp = await createImageBitmap(file);
  drawPhoto(bmp, [], bmp.width, bmp.height);
  try {
    const lm = await getLandmarker();
    status("Finding body landmarks…");
    const res = lm.detect(bmp);
    if (!res.landmarks.length) { status("No person found. Try a full-body photo in good light."); return; }
    const L = res.landmarks[0];
    const low = MP_INDEX.filter((i) => (L[i].visibility ?? 1) < 0.3);
    const kps = MP_INDEX.map((i) => [L[i].x * bmp.width, L[i].y * bmp.height]);
    await usePose(kps, bmp, bmp.width, bmp.height);
    if (low.length > 2) status("Pose ready, but some joints are hidden or out of frame, so answers may be off.");
  } catch (e) {
    status("Could not run the pose estimator: " + e.message);
  }
}

async function main() {
  for (const [task, [label]] of Object.entries(QUESTIONS)) {
    const b = Object.assign(document.createElement("button"), { className: "chip", textContent: label });
    b.dataset.task = task;
    b.onclick = () => ask(task);
    $("questions").append(b);
  }
  $("askform").onsubmit = (e) => {
    e.preventDefault();
    const q = $("ask").value.toLowerCase();
    ask((ROUTES.find(([re]) => re.test(q)) || [null, "describe"])[1]);
  };
  $("file").onchange = (e) => e.target.files[0] && onPhoto(e.target.files[0]);

  try {
    await loadModel();
  } catch (e) {
    status("Could not load the model: " + e.message);
    return;
  }
  const demos = await (await fetch("model/demo_poses.json")).json();
  for (const [name, kps] of Object.entries(demos)) {
    const b = Object.assign(document.createElement("button"), { className: "chip", textContent: name });
    b.onclick = () => { $("mirror").checked = false; usePose(kps, null, 1, 1); };
    $("demos").append(b);
  }
  status("Model loaded. Take a photo or pick an example.");
  usePose(Object.values(demos)[1], null, 1, 1);
}

main();
