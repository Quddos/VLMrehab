// GET /api/v1/health
import { cors, send } from "../_lib/http.js";
import {
  HAND_MODEL_VERSION, HAND_QUESTIONS, MODEL_VERSION, QUESTIONS, WRIST_MODEL_VERSION, WRIST_QUESTIONS,
} from "../../web/core.mjs";

export default function handler(req, res) {
  cors(res);
  send(res, 200, {
    ok: true,
    models: {
      body: { version: MODEL_VERSION, input: "joints | mediapipe", tasks: Object.keys(QUESTIONS) },
      hand: { version: HAND_MODEL_VERSION, input: "hand (21 landmarks)", tasks: Object.keys(HAND_QUESTIONS) },
      wrist: {
        version: WRIST_MODEL_VERSION, input: "hand (21 landmarks) + elbow + handedness",
        select: 'model: "wrist" or a question mentioning the wrist', tasks: Object.keys(WRIST_QUESTIONS),
      },
    },
  });
}
