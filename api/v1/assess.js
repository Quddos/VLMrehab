// POST /api/v1/assess  - run RehabVLM on a body pose or a hand.
//
// Headers: Authorization: Bearer rvlm_...
// Body (body model):
//   { joints: { nose: [x, y], l_shoulder: [...], ... } | mediapipe: [[x, y], ... 33],
//     space?: "image" | "unity", question?: "...", task?: "shoulders", tasks?: [...] }
// Body (hand model):
//   { hand: [[x, y, z], ... 21 landmarks], handedness?: "left" | "right",
//     space?: "image" | "unity", question?: "...", task?: "fist", tasks?: [...] }
import { cors, fail, send, rateLimited } from "../_lib/http.js";
import { bearer, verifyKey } from "../_lib/keys.js";
import { getModel } from "../_lib/model.js";
import {
  HAND_MODEL_VERSION, HAND_QUESTIONS, MODEL_VERSION, QUESTIONS, encodeHand, encodePose, generate,
  handFromInput, keypointsFromInput, measure, measureHand, routeQuestion,
} from "../../web/core.mjs";

const round1 = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v === null ? null : Math.round(v * 10) / 10]));

export default async function handler(req, res) {
  cors(res);
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return fail(res, 405, "method_not_allowed", "Use POST.");

  let keyId;
  try {
    keyId = verifyKey(bearer(req));
  } catch (e) {
    return fail(res, 500, "server_error", e.message);
  }
  if (!keyId) return fail(res, 401, "invalid_api_key", "Missing or invalid API key. Get one at /developers.html.");
  if (rateLimited("assess:" + keyId, 60, 60 * 1000)) {
    return fail(res, 429, "rate_limited", "Limit is 60 requests per minute per key.");
  }

  let body;
  try {
    body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  } catch {
    return fail(res, 400, "bad_request", "Body must be JSON.");
  }
  const domain = body.hand ? "hand" : "body";
  let points;
  try {
    points = domain === "hand" ? handFromInput(body) : keypointsFromInput(body);
  } catch (e) {
    return fail(res, 400, "bad_request", e.message);
  }
  const questions = domain === "hand" ? HAND_QUESTIONS : QUESTIONS;
  let tasks = Array.isArray(body.tasks) ? body.tasks : [body.task || routeQuestion(body.question, domain)];
  tasks = [...new Set(tasks.map((t) => routeQuestion(t, domain)))].slice(0, Object.keys(questions).length);

  const t0 = Date.now();
  const answers = {};
  try {
    const model = await getModel(domain);
    const { mem } = domain === "hand" ? await encodeHand(model, points) : await encodePose(model, points);
    for (const t of tasks) answers[t] = await generate(model, mem, t);
  } catch (e) {
    return fail(res, 500, "inference_error", e.message);
  }

  const result = {
    model: domain === "hand" ? HAND_MODEL_VERSION : MODEL_VERSION,
    answer: answers[tasks[0]],
    answers,
  };
  if (domain === "hand") {
    result.measurements = round1(measureHand(points));
    if (body.handedness) result.handedness = String(body.handedness).toLowerCase() === "left" ? "left" : "right";
  } else {
    result.angles_deg = round1(measure(points));
  }
  result.latency_ms = Date.now() - t0;
  result.disclaimer = "Research/education use only. Not a medical device.";
  send(res, 200, result);
}
