// GET /api/v1/health
import { cors, send } from "../_lib/http.js";
import { MODEL_VERSION, QUESTIONS } from "../../web/core.mjs";

export default function handler(req, res) {
  cors(res);
  send(res, 200, { ok: true, model: MODEL_VERSION, tasks: Object.keys(QUESTIONS) });
}
