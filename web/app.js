// RehabVLM browser app: photo -> MediaPipe pose -> skeleton raster -> tiny VLM (onnxruntime-web).
import {
  IMG, QUESTIONS, KP, MP_INDEX, SEGMENTS, SEGMENT_INDEX, mid, measure, routeQuestion,
  loadModel, encodePose, generate,
} from "./core.mjs";

const MP_VERSION = "0.10.14";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const ANGLE_LABELS = {
  r_shoulder_abd: "Right shoulder abduction", l_shoulder_abd: "Left shoulder abduction",
  r_elbow_flex: "Right elbow flexion", l_elbow_flex: "Left elbow flexion",
  r_hip_abd: "Right hip abduction", l_hip_abd: "Left hip abduction",
  trunk_lean: "Trunk lean (+ = to left)",
};

const $ = (id) => document.getElementById(id);
const status = (s) => { $("status").textContent = s; };

let model, landmarker;
let current = null;   // { kps, pixels, mem }
let busy = false, pending = null, runId = 0;

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
  if (!kps.length) return;
  const P = (p) => [ox + p[0] * s, oy + p[1] * s];
  const colors = ["#ef4444", "#22c55e", "#3b82f6"];
  const all = [...kps, mid(kps[1], kps[2]), mid(kps[7], kps[8])];
  ctx.lineWidth = 3; ctx.lineCap = "round";
  for (const [a, b, ch] of SEGMENTS) {
    const p = P(all[SEGMENT_INDEX[a]]), q = P(all[SEGMENT_INDEX[b]]);
    ctx.strokeStyle = colors[ch]; ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  for (const p of kps) { const [x, y] = P(p); ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); }
}

function showGeometry(kps) {
  $("geom").innerHTML = Object.entries(measure(kps))
    .map(([k, v]) => `<tr><td>${ANGLE_LABELS[k]}</td><td>${v.toFixed(0)}°</td></tr>`).join("");
}

// ---------- pipeline ----------
async function usePose(kps, source, w, h) {
  if ($("mirror").checked) {
    // a mirrored selfie swaps the subject's sides: flip x and swap left/right labels
    kps = kps.map((p) => [w - p[0], p[1]]);
    const swap = (i, j) => { [kps[i], kps[j]] = [kps[j], kps[i]]; };
    for (let i = 1; i < KP.length; i += 2) swap(i, i + 1);
  }
  const { pixels, mem } = await encodePose(model, kps);
  drawPhoto(source, kps, w, h);
  drawSkeleton(pixels);
  showGeometry(kps);
  current = { kps, pixels, mem };
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
    const done = await generate(model, current.mem, task, {
      onToken: (s) => { a.textContent = s; },
      shouldStop: () => id !== runId,   // superseded by a newer question or pose
    });
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
    ask(routeQuestion($("ask").value));
  };
  $("file").onchange = (e) => e.target.files[0] && onPhoto(e.target.files[0]);

  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const vocab = await (await fetch("model/vocab.json")).json();
    model = await loadModel(ort, vocab, async (f) => `model/${f}`);
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
