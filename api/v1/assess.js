// POST /api/v1/assess  - run RehabVLM on a body pose.
//
// Headers: Authorization: Bearer rvlm_...
// Body:    { joints: { nose: [x, y], l_shoulder: [...], ... }, space?: "image" | "unity",
//            question?: "how is my left shoulder?" | task?: "shoulders",
//            tasks?: ["shoulders", "exercise"] }
//      or  { mediapipe: [[x, y], ... 33 landmarks], ... }
import { cors, fail, send, rateLimited } from "../_lib/http.js";
import { bearer, verifyKey } from "../_lib/keys.js";
import { getModel } from "../_lib/model.js";
import {
  MODEL_VERSION, QUESTIONS, encodePose, generate, keypointsFromInput, measure, routeQuestion,
} from "../../web/core.mjs";

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

  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  let kps;
  try {
    kps = keypointsFromInput(body);
  } catch (e) {
    return fail(res, 400, "bad_request", e.message);
  }
  let tasks = Array.isArray(body.tasks) ? body.tasks : [body.task || routeQuestion(body.question)];
  tasks = [...new Set(tasks.map(routeQuestion))].slice(0, Object.keys(QUESTIONS).length);

  const t0 = Date.now();
  const answers = {};
  try {
    const model = await getModel();
    const { mem } = await encodePose(model, kps);
    for (const t of tasks) answers[t] = await generate(model, mem, t);
  } catch (e) {
    return fail(res, 500, "inference_error", e.message);
  }
  const angles = Object.fromEntries(Object.entries(measure(kps)).map(([k, v]) => [k, Math.round(v * 10) / 10]));

  send(res, 200, {
    model: MODEL_VERSION,
    answer: answers[tasks[0]],
    answers,
    angles_deg: angles,
    latency_ms: Date.now() - t0,
    disclaimer: "Research/education use only. Not a medical device.",
  });
}
