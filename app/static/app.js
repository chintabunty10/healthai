// All dynamic text is inserted with textContent (never innerHTML) to avoid XSS.
const form = document.getElementById("symptom-form");
const input = document.getElementById("symptoms");
const btn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const urgentEl = document.getElementById("urgent");
const results = document.getElementById("results");
const imagesInput = document.getElementById("images");
const MAX_IMAGES = 3, MAX_BYTES = 5 * 1024 * 1024;
const list = document.getElementById("conditions");
let lastSymptoms = [];

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = isError ? "error" : "";
}

function errorText(data) {
  const d = data && data.detail;
  if (typeof d === "string") return d;
  if (d && d.message) return d.message;
  if (Array.isArray(d) && d[0] && d[0].msg) return d[0].msg;
  return "Something went wrong. Please try again.";
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function renderCondition(c) {
  const li = el("li", "card");
  const row = el("div", "row");
  row.append(el("span", "name", c.name), el("span", "badge", Math.round(c.probability * 100) + "% match"));
  const bar = el("div", "bar");
  const fill = el("span");
  fill.style.width = Math.max(2, Math.round(c.probability * 100)) + "%";
  bar.append(fill);
  li.append(row, bar);
  if (c.description) li.append(el("p", "meta", c.description));

  const more = el("button", "secondary", "Explain this condition");
  more.type = "button";
  const out = el("div", "explanation");
  out.hidden = true;
  more.addEventListener("click", async () => {
    more.disabled = true;
    more.textContent = "Loading…";
    try {
      const r = await fetch("/api/explain", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ condition: c.name, symptoms: lastSymptoms }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(errorText(data));
      out.textContent = data.explanation;
      out.append(el("div", "badge", data.source === "llm" ? (data.cached ? "AI explanation (cached)" : "AI explanation") : "Basic description (AI unavailable)"));
      out.hidden = false;
      more.remove();
    } catch (e) {
      more.disabled = false;
      more.textContent = "Explain this condition";
      setStatus(e.message, true);
    }
  });
  li.append(more, out);
  return li;
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = input.value.trim();
  const files = Array.from(imagesInput.files);
  if (!text && files.length === 0) { setStatus("Enter at least one symptom or add a photo.", true); return; }
  if (files.length > MAX_IMAGES) { setStatus("Please choose at most " + MAX_IMAGES + " photos.", true); return; }
  if (files.some((f) => f.size > MAX_BYTES)) { setStatus("Each photo must be under 5 MB.", true); return; }
  btn.disabled = true;
  setStatus(files.length ? "Analysing photo(s) and symptoms…" : "Analysing…");
  urgentEl.hidden = true;
  try {
    let r;
    if (files.length) {                       // photos present -> multipart image endpoint
      const fd = new FormData();
      fd.append("symptoms", text);
      files.forEach((f) => fd.append("images", f));
      r = await fetch("/api/predict/image", { method: "POST", body: fd });
    } else {                                  // text only -> JSON endpoint
      r = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: text }),
      });
    }
    const data = await r.json();
    if (!r.ok) throw new Error(errorText(data));
    lastSymptoms = data.matched_symptoms;
    document.getElementById("matched").textContent =
      "Understood: " + data.matched_symptoms.join(", ").replaceAll("_", " ") +
      (data.unmatched_symptoms.length ? " · Not recognised: " + data.unmatched_symptoms.join(", ") : "") +
      (data.cached ? " · (cached result)" : "");
    const imgDesc = document.getElementById("image-desc");
    imgDesc.hidden = !data.image_description;
    imgDesc.textContent = data.image_description ? "From your photo(s): " + data.image_description : "";
    const warn = document.getElementById("warnings");
    warn.hidden = !(data.warnings && data.warnings.length);
    warn.textContent = (data.warnings || []).join(" ");
    list.replaceChildren(...data.conditions.map(renderCondition));
    document.getElementById("disclaimer").textContent = data.disclaimer;
    if (data.urgent_warning) { urgentEl.textContent = data.urgent_warning; urgentEl.hidden = false; }
    results.hidden = false;
    setStatus("");
  } catch (e) {
    results.hidden = true;
    setStatus(e.message, true);
  } finally {
    btn.disabled = false;
  }
});
