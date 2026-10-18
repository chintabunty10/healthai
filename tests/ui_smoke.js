// Minimal DOM stand-in to execute the REAL app.js and exercise its main flows (no browser available offline).
const fs = require("fs");
class Node_ {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.handlers = {}; this.style = {}; this.dataset = {};
    this._text = ""; this.className = ""; this.hidden = false; this.disabled = false; this.value = ""; this.files = []; this.classList = { add: () => {}, remove: () => {} }; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  append(...n) { n.forEach((x) => this.children.push(typeof x === "string" ? Object.assign(new Node_("#text"), { _text: x }) : x)); }
  replaceChildren(...n) { this.children = []; this._text = ""; this.append(...n); }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(t, f) { (this.handlers[t] = this.handlers[t] || []).push(f); }
  async fire(t, ev = {}) { for (const f of this.handlers[t] || []) await f({ preventDefault() {}, ...ev }); }
  focus() {} scrollIntoView() {} remove() {}
  walk(fn) { fn(this); this.children.forEach((c) => c.walk && c.walk(fn)); }
}
const registry = {};
global.document = {
  getElementById: (id) => (registry[id] = registry[id] || new Node_("#" + id)),
  createElement: (t) => new Node_(t),
  querySelectorAll: () => [],
};
global.URL = { createObjectURL: () => "blob:x", revokeObjectURL() {} };
global.FormData = class { constructor() { this.d = []; } append(k, v) { this.d.push([k, v]); } };
global.AbortController = class { constructor() { this.signal = {}; } abort() {} };
const calls = [];
const respond = {
  "/api/symptoms": { symptoms: ["cough", "skin rash"] },
  "/api/predict": { matched_symptoms: ["chest_pain", "cough"], unmatched_symptoms: ["zzz"], cached: false, disclaimer: "Not a diagnosis.",
    urgent_warning: "Seek urgent care.", image_description: null, warnings: [],
    conditions: [{ name: "Heart attack", probability: 0.6, description: "d", precautions: ["Call ambulance"] }, { name: "Pneumonia", probability: 0.004, description: "", precautions: [] }] },
  "/api/predict/image": { matched_symptoms: ["skin_rash"], unmatched_symptoms: [], cached: true, disclaimer: "x", urgent_warning: null,
    image_description: "Red patches", warnings: ["Image analysis was unavailable"], conditions: [{ name: "Fungal infection", probability: 0.79, description: "", precautions: [] }] },
  "/api/explain": { condition: "x", explanation: "Plain words.", cached: false, source: "llm" },
};
global.fetch = async (path, init = {}) => { calls.push([path, init.method || "GET", init.body]);
  return { ok: true, status: 200, json: async () => respond[path] }; };
eval(fs.readFileSync("app/static/app.js", "utf8"));
const $ = (id) => registry[id];
const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } console.log("ok  ", m); };
const texts = (n) => { const out = []; n.walk((x) => x._text && out.push(x._text)); return out.join(" | "); };
(async () => {
  await new Promise((r) => setTimeout(r, 10));
  assert(calls.some((c) => c[0] === "/api/symptoms"), "loads symptom suggestions on start");
  // add chips through the input
  $("symptom-input").value = "chest pain";
  await $("symptom-input").fire("keydown", { key: "Enter" });
  $("symptom-input").value = "Cough, cough";
  await $("symptom-input").fire("keydown", { key: "," });
  assert($("chips").children.length === 2, "two unique chips (duplicate ignored, case-normalized)");
  // submit text-only
  await $("symptom-form").fire("submit");
  assert(calls.some((c) => c[0] === "/api/predict" && JSON.parse(c[2]).symptoms === "chest pain, cough"), "text-only submit uses JSON /api/predict");
  let t = texts($("results-body"));
  assert(t.includes("Seek urgent care"), "urgent banner shown");
  assert(t.includes("Heart attack") && t.includes("60%") && t.includes("<1%"), "ranked cards with percentages (<1% handled)");
  assert(t.includes("Not recognised: zzz") && t.includes("General precautions") && t.includes("Call ambulance"), "unmatched notice + precautions rendered");
  // explain one
  let explainBtn; $("results-body").walk((n) => { if (n.className === "secondary" && n._text === "Explain this condition" && !explainBtn) explainBtn = n; });
  await explainBtn.fire("click");
  assert(texts($("results-body")).includes("Plain words.") && texts($("results-body")).includes("AI explanation"), "explain button renders LLM text + badge");
  // explain all
  let allBtn; $("results-body").walk((n) => { if (n._text === "Explain all") allBtn = n; });
  await allBtn.fire("click");
  assert(calls.filter((c) => c[0] === "/api/explain").length === 2, "explain-all only calls cards not yet explained (2 total)");
  // photo path
  $("images").files = [{ name: "a.jpg", type: "image/jpeg", size: 1000 }];
  await $("images").fire("change");
  assert($("thumbs").children.length === 1, "photo thumbnail added");
  await $("symptom-form").fire("submit");
  assert(calls.some((c) => c[0] === "/api/predict/image" && c[1] === "POST"), "photos present -> multipart /api/predict/image");
  t = texts($("results-body"));
  assert(t.includes("From your photo(s): Red patches") && t.includes("Image analysis was unavailable") && t.includes("from cache"), "image description, warning and cache note shown");
  // validation
  $("images").files = [{ name: "x.gif", type: "image/gif", size: 10 }];
  await $("images").fire("change");
  assert($("status").className === "error" && $("status").textContent.includes("not a JPEG"), "rejects non-image files client-side");
  // reset
  await $("reset-btn").fire("click");
  assert($("chips").children.length === 0 && texts($("results-body")).includes("Your results will appear here"), "clear restores the empty state");
  // empty submit
  await $("symptom-form").fire("submit");
  assert($("status").className === "error" && $("status").textContent.includes("at least one"), "empty submit shows a friendly error");
  console.log("\nALL UI FLOW CHECKS PASSED");
})().catch((e) => { console.error("ERROR", e); process.exit(1); });
