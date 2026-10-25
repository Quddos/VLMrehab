// POST /api/keys -> { key }   Issues a free API key.
import { cors, fail, send, rateLimited, clientIp } from "./_lib/http.js";
import { issueKey } from "./_lib/keys.js";

export default function handler(req, res) {
  cors(res);
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return fail(res, 405, "method_not_allowed", "Use POST.");
  if (rateLimited("keys:" + clientIp(req), 5, 60 * 60 * 1000)) {
    return fail(res, 429, "rate_limited", "Too many keys requested from this network. Try again later.");
  }
  try {
    const { id, key } = issueKey();
    send(res, 201, {
      key, key_id: id,
      limits: { requests_per_minute: 60 },
      docs: "https://rehabvlm.vercel.app/developers.html",
    });
  } catch (e) {
    fail(res, 500, "server_error", e.message);
  }
}
