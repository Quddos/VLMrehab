// Stateless API keys: rvlm_<id>.<hmac(id)>. Any key whose signature verifies against
// RVLM_KEY_SECRET is valid, so no database is needed. Revoke a key by adding its id to
// the comma-separated RVLM_REVOKED_KEYS environment variable.
import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";

function secret() {
  const s = process.env.RVLM_KEY_SECRET;
  if (!s || s.length < 32) throw new Error("RVLM_KEY_SECRET is not configured");
  return s;
}

const sign = (id) => createHmac("sha256", secret()).update(id).digest("base64url").slice(0, 32);

export function issueKey() {
  const id = randomBytes(12).toString("hex");
  return { id, key: `rvlm_${id}.${sign(id)}` };
}

/** Returns the key id when valid, otherwise null. */
export function verifyKey(key) {
  const m = /^rvlm_([0-9a-f]{24})\.([A-Za-z0-9_-]{32})$/.exec(String(key || ""));
  if (!m) return null;
  const [, id, sig] = m;
  const expected = Buffer.from(sign(id));
  if (!timingSafeEqual(expected, Buffer.from(sig))) return null;
  const revoked = (process.env.RVLM_REVOKED_KEYS || "").split(",").map((s) => s.trim());
  return revoked.includes(id) ? null : id;
}

export function bearer(req) {
  const h = String(req.headers.authorization || "");
  return h.startsWith("Bearer ") ? h.slice(7).trim() : String(req.headers["x-api-key"] || "");
}
