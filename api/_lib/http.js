// Small helpers shared by the API routes.

export function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Authorization, Content-Type");
}

export function send(res, status, body) {
  res.status(status).json(body);
}

export function fail(res, status, code, message) {
  send(res, status, { error: { code, message } });
}

// Best-effort fixed-window limiter. State lives per function instance, so it smooths bursts
// rather than enforcing a global quota; use a shared store for hard limits.
const windows = new Map();

export function rateLimited(key, limit, windowMs) {
  const now = Date.now();
  let w = windows.get(key);
  if (!w || now - w.start > windowMs) {
    w = { start: now, count: 0 };
    windows.set(key, w);
    if (windows.size > 10000) windows.clear();
  }
  w.count += 1;
  return w.count > limit;
}

export function clientIp(req) {
  return String(req.headers["x-forwarded-for"] || "").split(",")[0].trim() || "unknown";
}
