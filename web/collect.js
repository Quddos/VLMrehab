// Clinician data collection: tracker landmarks + goniometer labels -> JSONL records for finetune.py.
// Only landmarks and numbers are kept; photos never leave the canvas.
import { MP_INDEX, HAND_SEGMENTS, measure, measureHand, measureWrist } from "./core.mjs";

const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const STORE = "rehabvlm-collect-records";

const MODES = {
  body: {
    instructions: "Full body in frame, facing the camera, at the end of the movement you are measuring.",
    labels: {
      r_shoulder_abd: "Right shoulder abduction", l_shoulder_abd: "Left shoulder abduction",
      r_elbow_flex: "Right elbow flexion", l_elbow_flex: "Left elbow flexion",
      r_hip_abd: "Right hip abduction", l_hip_abd: "Left hip abduction",
      trunk_lean: "Trunk lean (+ towards subject's left)",
    },
  },
  hand: {
    instructions: "One hand, all fingers visible. Finger flexion = MCP + PIP + DIP (thumb: MCP + IP), summed.",
    labels: {
      thumb_flex: "Thumb flexion (MCP + IP)", index_flex: "Index flexion (sum)", middle_flex: "Middle flexion (sum)",
      ring_flex: "Ring flexion (sum)", pinky_flex: "Pinky flexion (sum)",
    },
  },
  wrist: {
    instructions: "Hand and forearm up to the elbow in frame. Flexion positive, extension negative; radial positive, ulnar negative.",
    labels: { wrist_flex: "Wrist flexion (+) / extension (−)", wrist_dev: "Radial (+) / ulnar (−) deviation" },
  },
};

const $ = (id) => document.getElementById(id);
const status = (s) => { $("status").textContent = s; };
let mode = "body", pending = null, detectors = {};
let records = [];
try { records = JSON.parse(localStorage.getItem(STORE) || "[]"); } catch { records = []; }

function persist() {
  try { localStorage.setItem(STORE, JSON.stringify(records)); } catch { /* private mode: memory only */ }
  $("count").textContent = `${records.length} record${records.length === 1 ? "" : "s"} · ` +
    `${new Set(records.map((r) => r.subject)).size} subject(s)`;
  $("download").disabled = !records.length;
}

async function detector(kind) {
  if (!detectors[kind]) {
    detectors[kind] = (async () => {
      const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      return kind === "pose"
        ? vision.PoseLandmarker.createFromOptions(fileset, { runningMode: "IMAGE", numPoses: 1, baseOptions: { modelAssetPath: POSE_MODEL } })
        : vision.HandLandmarker.createFromOptions(fileset, { runningMode: "IMAGE", numHands: 1, baseOptions: { modelAssetPath: HAND_MODEL } });
    })();
    detectors[kind].catch(() => { delete detectors[kind]; });
  }
  return detectors[kind];
}

function draw(bmp, pts2d, lines) {
  const c = $("photo"), ctx = c.getContext("2d");
  const s = Math.min(c.width / bmp.width, c.height / bmp.height);
  const ox = (c.width - bmp.width * s) / 2, oy = (c.height - bmp.height * s) / 2;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(bmp, ox, oy, bmp.width * s, bmp.height * s);
  const P = (p) => [ox + p[0] * s, oy + p[1] * s];
  ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 3;
  for (const [a, b] of lines) { ctx.beginPath(); ctx.moveTo(...P(pts2d[a])); ctx.lineTo(...P(pts2d[b])); ctx.stroke(); }
  ctx.fillStyle = "#fff";
  for (const p of pts2d) { const [x, y] = P(p); ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); }
}

const unit3 = (v) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };

async function onPhoto(file) {
  const bmp = await createImageBitmap(file);
  pending = null;
  $("add").disabled = true;
  try {
    status("Loading tracker…");
    if (mode === "body") {
      const res = (await detector("pose")).detect(bmp);
      if (!res.landmarks.length) return status("No person found.");
      const mp = res.landmarks[0].map((p) => [p.x * bmp.width, p.y * bmp.height]);
      draw(bmp, MP_INDEX.map((i) => mp[i]), [[1, 3], [3, 5], [2, 4], [4, 6], [1, 2], [7, 8], [1, 7], [2, 8], [7, 9], [9, 11], [8, 10], [10, 12]]);
      pending = { record: { mediapipe: mp.map((p) => p.map((x) => +x.toFixed(2))) }, measured: measure(MP_INDEX.map((i) => mp[i])) };
    } else {
      const res = (await detector("hand")).detect(bmp);
      if (!res.landmarks.length) return status("No hand found.");
      const world = res.worldLandmarks[0].map((p) => [p.x, p.y, p.z]);
      draw(bmp, res.landmarks[0].map((p) => [p.x * bmp.width, p.y * bmp.height]), HAND_SEGMENTS);
      const handedness = $("handedness").value;
      const record = { hand: world.map((p) => p.map((x) => +x.toFixed(5))), handedness };
      let measured = measureHand(world);
      if (mode === "wrist") {
        // forearm direction from body pose (camera-aligned axes like the hand's world landmarks)
        const pose = (await detector("pose")).detect(bmp);
        if (!pose.worldLandmarks.length) return status("No forearm found. Keep the elbow in frame.");
        const W = pose.worldLandmarks[0], [wi, ei] = handedness === "left" ? [15, 13] : [16, 14];
        const fore = unit3([W[wi].x - W[ei].x, W[wi].y - W[ei].y, W[wi].z - W[ei].z]);
        const elbow = world[0].map((x, d) => x - fore[d] * 0.25);
        record.elbow = elbow.map((x) => +x.toFixed(5));
        measured = measureWrist(world, elbow, handedness);
      }
      pending = { record, measured };
    }
    showLabels();
    status("Landmarks ready. Enter your goniometer readings and save.");
    $("add").disabled = false;
  } catch (e) {
    status("Tracker error: " + e.message);
  }
}

function showLabels() {
  const m = pending ? pending.measured : {};
  $("labels").innerHTML = Object.entries(MODES[mode].labels).map(([k, label]) =>
    `<label>${label}<span class="tracker">${m[k] == null ? "" : `tracker ${m[k].toFixed(0)}°`}</span>
       <input type="number" inputmode="decimal" step="1" data-key="${k}"></label>`).join("");
}

function save() {
  const subject = $("subject").value.trim();
  if (!$("consent").checked) return status("Tick the consent box first.");
  if (!subject) return status("Enter a pseudonymous subject ID (no names).");
  if (!pending) return status("Add a photo first.");
  const labels = {};
  document.querySelectorAll("#labels input").forEach((i) => {
    if (i.value.trim() !== "" && Number.isFinite(+i.value)) labels[i.dataset.key] = +i.value;
  });
  if (!Object.keys(labels).length) return status("Enter at least one measurement.");
  records.push({
    domain: mode, subject, consent: true, source: "goniometer", space: "image",
    recorded: new Date().toISOString().slice(0, 10), ...pending.record, labels,
  });
  persist();
  pending = null;
  $("add").disabled = true;
  showLabels();
  status("Saved. Take the next photo.");
}

function setMode(m) {
  mode = m;
  pending = null;
  $("add").disabled = true;
  document.querySelectorAll(".mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  $("handed-row").hidden = m === "body";
  $("instructions").textContent = MODES[m].instructions;
  showLabels();
}

document.querySelectorAll(".mode button").forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
$("file").onchange = (e) => { if (e.target.files[0]) onPhoto(e.target.files[0]); e.target.value = ""; };
$("add").onclick = save;
$("download").onclick = () => {
  const blob = new Blob([records.map((r) => JSON.stringify(r)).join("\n") + "\n"], { type: "application/x-ndjson" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: `rehabvlm-records-${new Date().toISOString().slice(0, 10)}.jsonl`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
};
$("clear").onclick = () => {
  if (records.length && !confirm(`Delete ${records.length} unsaved record(s) from this browser?`)) return;
  records = [];
  persist();
};
setMode("body");
persist();
