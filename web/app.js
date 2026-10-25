// RehabVLM browser app: photo -> MediaPipe (pose or hand) -> skeleton raster -> tiny VLM (onnxruntime-web).
import {
  IMG, QUESTIONS, HAND_QUESTIONS, KP, MP_INDEX, SEGMENTS, SEGMENT_INDEX, HAND_SEGMENTS, mid, measure,
  measureHand, routeQuestion, loadModel, encodePose, encodeHand, generate,
} from "./core.mjs";

const MP_VERSION = "0.10.14";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const MODES = {
  body: {
    title: "1 · Pose",
    instructions: "Stand facing the camera with your whole body in frame, then raise both arms out to the side as far as you comfortably can.",
    placeholder: "e.g. how is my right shoulder?",
    questions: QUESTIONS, dir: "model", demos: "demo_poses.json", firstDemo: 1,
    labels: {
      r_shoulder_abd: "Right shoulder abduction", l_shoulder_abd: "Left shoulder abduction",
      r_elbow_flex: "Right elbow flexion", l_elbow_flex: "Left elbow flexion",
      r_hip_abd: "Right hip abduction", l_hip_abd: "Left hip abduction",
      trunk_lean: "Trunk lean (+ = to left)",
    },
  },
  hand: {
    title: "1 · Hand",
    instructions: "Hold one hand up to the camera with all fingers visible, then make a fist, pinch, or spread your fingers as far as you can.",
    placeholder: "e.g. which finger is stiffest?",
    questions: HAND_QUESTIONS, dir: "model/hand", demos: "demo_hands.json", firstDemo: 2,
    labels: {
      thumb_flex: "Thumb flexion", index_flex: "Index flexion", middle_flex: "Middle flexion",
      ring_flex: "Ring flexion", pinky_flex: "Pinky flexion", spread: "Index–pinky spread",
      pinch_gap: "Thumb–index gap (% hand width)",
    },
  },
};

const $ = (id) => document.getElementById(id);
const status = (s) => { $("status").textContent = s; };

let mode = "body";
const models = {}, detectors = {}, demoSets = {};
let current = null;   // { mode, points, mem }
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

// Draws the photo (if any) and 2-D landmarks; lines = [[a, b, colour], ...] over pts.
function drawPhoto(source, pts, lines, w, h) {
  const c = $("photo"), ctx = c.getContext("2d");
  const s = Math.min(c.width / w, c.height / h);
  const ox = (c.width - w * s) / 2, oy = (c.height - h * s) / 2;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
  if (source) ctx.drawImage(source, ox, oy, w * s, h * s);
  if (!pts.length) return;
  const P = (p) => [ox + p[0] * s, oy + p[1] * s];
  ctx.lineWidth = 3; ctx.lineCap = "round";
  for (const [a, b, colour] of lines) {
    ctx.strokeStyle = colour; ctx.beginPath(); ctx.moveTo(...P(pts[a])); ctx.lineTo(...P(pts[b])); ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  for (const p of pts) { const [x, y] = P(p); ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); }
}

function bodyLines(kps) {
  const colours = ["#ef4444", "#22c55e", "#3b82f6"];
  const pts = [...kps, mid(kps[1], kps[2]), mid(kps[7], kps[8])];
  return [pts, SEGMENTS.map(([a, b, ch]) => [SEGMENT_INDEX[a], SEGMENT_INDEX[b], colours[ch]])];
}

const handLines = () => HAND_SEGMENTS.map(([a, b]) => [a, b, "#f59e0b"]);

// Fit 2-D points into a unit square for drawing when there is no photo.
function fitUnit(pts) {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys);
  const span = Math.max(Math.max(...xs) - x0, Math.max(...ys) - y0) || 1;
  return pts.map((p) => [0.1 + 0.8 * (p[0] - x0) / span, 0.1 + 0.8 * (p[1] - y0) / span]);
}

function showGeometry(values) {
  const labels = MODES[mode].labels;
  $("geom").innerHTML = Object.entries(values)
    .map(([k, v]) => `<tr><td>${labels[k]}</td><td>${v === null ? "n/a" : v.toFixed(0) + (k === "pinch_gap" ? "%" : "°")}</td></tr>`)
    .join("");
}

// ---------- models ----------
async function getModel(m) {
  if (!models[m]) {
    const { dir, questions } = MODES[m];
    models[m] = (async () => {
      const vocab = await (await fetch(`${dir}/vocab.json`)).json();
      return loadModel(ort, vocab, async (f) => `${dir}/${f}`, questions);
    })();
    models[m].catch(() => { delete models[m]; });
  }
  return models[m];
}

async function getDetector(m) {
  if (!detectors[m]) {
    status(`Loading ${m} tracker (first time only)…`);
    detectors[m] = (async () => {
      const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      const opts = { runningMode: "IMAGE" };
      return m === "body"
        ? vision.PoseLandmarker.createFromOptions(fileset, { ...opts, numPoses: 1, baseOptions: { modelAssetPath: POSE_MODEL } })
        : vision.HandLandmarker.createFromOptions(fileset, { ...opts, numHands: 1, baseOptions: { modelAssetPath: HAND_MODEL } });
    })();
    detectors[m].catch(() => { delete detectors[m]; });
  }
  return detectors[m];
}

// ---------- pipeline ----------
async function useBody(kps, source, w, h) {
  if ($("mirror").checked) {
    // a mirrored selfie swaps the subject's sides: flip x and swap left/right labels
    kps = kps.map((p) => [w - p[0], p[1]]);
    for (let i = 1; i < KP.length; i += 2) [kps[i], kps[i + 1]] = [kps[i + 1], kps[i]];
  }
  const model = await getModel("body");
  const { pixels, mem } = await encodePose(model, kps);
  const [pts, lines] = bodyLines(kps);
  drawPhoto(source, pts, lines, w, h);
  drawSkeleton(pixels);
  showGeometry(measure(kps));
  current = { mode: "body", mem };
  status("Pose ready. Ask a question.");
  ask("describe");
}

// landmarks3d: 21 x [x, y, z]; image2d: 21 x [x, y] in photo pixels (or null for demos)
async function useHand(landmarks3d, source, image2d, w, h) {
  const model = await getModel("hand");
  const { pixels, mem } = await encodeHand(model, landmarks3d);
  drawPhoto(source, image2d || fitUnit(landmarks3d), handLines(), image2d ? w : 1, image2d ? h : 1);
  drawSkeleton(pixels);
  showGeometry(measureHand(landmarks3d));
  current = { mode: "hand", mem };
  status("Hand ready. Ask a question.");
  ask("describe");
}

async function ask(task) {
  if (!current) { status("Add a photo or pick an example first."); return; }
  const id = ++runId;
  if (busy) { pending = task; return; }
  busy = true;
  const questions = MODES[current.mode].questions;
  document.querySelectorAll("#questions .chip").forEach((b) => b.classList.toggle("on", b.dataset.task === task));
  const box = $("answer");
  box.innerHTML = `<span class="q">${questions[task][0]}</span><span class="a"></span>`;
  const a = box.querySelector(".a");
  const t0 = performance.now();
  try {
    const done = await generate(await getModel(current.mode), current.mem, task, {
      onToken: (s) => { a.textContent = s; },
      shouldStop: () => id !== runId,   // superseded by a newer question or input
    });
    if (done !== null) status(`Answered in ${((performance.now() - t0) / 1000).toFixed(2)} s, on your device.`);
  } catch (e) {
    a.textContent = "Error: " + e.message;
  } finally {
    busy = false;
  }
  if (pending) { const next = pending; pending = null; ask(next); }
}

async function onPhoto(file) {
  const bmp = await createImageBitmap(file);
  drawPhoto(bmp, [], [], bmp.width, bmp.height);
  try {
    const detector = await getDetector(mode);
    status(mode === "body" ? "Finding body landmarks…" : "Finding hand landmarks…");
    const res = detector.detect(bmp);
    if (mode === "body") {
      if (!res.landmarks.length) { status("No person found. Try a full-body photo in good light."); return; }
      const L = res.landmarks[0];
      const low = MP_INDEX.filter((i) => (L[i].visibility ?? 1) < 0.3);
      await useBody(MP_INDEX.map((i) => [L[i].x * bmp.width, L[i].y * bmp.height]), bmp, bmp.width, bmp.height);
      if (low.length > 2) status("Pose ready, but some joints are hidden or out of frame, so answers may be off.");
    } else {
      if (!res.landmarks.length) { status("No hand found. Hold one hand up, fingers visible, in good light."); return; }
      const world = res.worldLandmarks[0].map((p) => [p.x, p.y, p.z]);
      const image = res.landmarks[0].map((p) => [p.x * bmp.width, p.y * bmp.height]);
      await useHand(world, bmp, image, bmp.width, bmp.height);
    }
  } catch (e) {
    status("Could not run the tracker: " + e.message);
  }
}

async function setMode(m) {
  mode = m;
  current = null;
  runId++;
  const cfg = MODES[m];
  $("mode-body").classList.toggle("on", m === "body");
  $("mode-hand").classList.toggle("on", m === "hand");
  $("step1").textContent = cfg.title;
  $("instructions").textContent = cfg.instructions;
  $("ask").placeholder = cfg.placeholder;
  $("mirror").parentElement.style.display = m === "body" ? "" : "none";
  $("seecap").textContent = m === "body" ? "What the VLM sees (96×96)" : "What the VLM sees: front, side and top views";
  $("answer").innerHTML = "";
  $("geom").innerHTML = "";

  $("questions").innerHTML = "";
  for (const [task, [label]] of Object.entries(cfg.questions)) {
    const b = Object.assign(document.createElement("button"), { className: "chip", textContent: label });
    b.dataset.task = task;
    b.onclick = () => ask(task);
    $("questions").append(b);
  }

  status("Loading model…");
  try {
    await getModel(m);
    demoSets[m] = demoSets[m] || await (await fetch(`${cfg.dir}/${cfg.demos}`)).json();
  } catch (e) {
    status("Could not load the model: " + e.message);
    return;
  }
  if (mode !== m) return;   // switched again while loading
  $("demos").innerHTML = "";
  const use = (pts) => (m === "body" ? useBody(pts, null, 1, 1) : useHand(pts, null, null));
  for (const [name, pts] of Object.entries(demoSets[m])) {
    const b = Object.assign(document.createElement("button"), { className: "chip", textContent: name });
    b.onclick = () => { $("mirror").checked = false; use(pts); };
    $("demos").append(b);
  }
  use(Object.values(demoSets[m])[cfg.firstDemo]);
}

function main() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  $("askform").onsubmit = (e) => {
    e.preventDefault();
    if (current) ask(routeQuestion($("ask").value, current.mode));
  };
  $("file").onchange = (e) => { if (e.target.files[0]) onPhoto(e.target.files[0]); e.target.value = ""; };
  $("mode-body").onclick = () => mode !== "body" && setMode("body");
  $("mode-hand").onclick = () => mode !== "hand" && setMode("hand");
  setMode(new URLSearchParams(location.search).get("mode") === "hand" ? "hand" : "body");
}

main();
