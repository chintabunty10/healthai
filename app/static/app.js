"use strict";
// HealthAI front end. All dynamic text is inserted with textContent (never innerHTML) to avoid XSS.
(() => {
  const MAX_SYMPTOMS = 20, MAX_LEN = 60, MAX_TEXT = 500, MAX_IMAGES = 3, MAX_BYTES = 5 * 1024 * 1024;
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const QUICK = ["high fever", "cough", "headache", "fatigue", "vomiting", "nausea", "chills", "skin rash", "abdominal pain", "dizziness"];
  const EXAMPLES = [["fever", "cough", "chills"], ["itching", "skin rash"], ["headache", "vomiting", "nausea"]];

  const $ = (id) => document.getElementById(id);
  const state = { symptoms: [], files: [], matched: [], busy: false, cards: [] };

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const norm = (s) => String(s).trim().toLowerCase().replace(/\s+/g, " ");
  const pct = (p) => (p < 0.01 ? "<1%" : Math.round(p * 100) + "%");

  function setStatus(msg, isError = false) {
    const s = $("status");
    s.textContent = msg;
    s.className = isError ? "error" : "";
  }

  function errorText(data, status) {
    const d = data && data.detail;
    if (typeof d === "string") return d;
    if (d && d.message) return d.message;
    if (Array.isArray(d) && d[0] && d[0].msg) return d[0].msg;
    return "Something went wrong (" + status + "). Please try again.";
  }

  async function api(path, options = {}) {
    const { timeout = 30000, ...init } = options;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeout);
    try {
      const r = await fetch(path, { ...init, signal: ctrl.signal });
      let data = null;
      try { data = await r.json(); } catch (_) { /* non-JSON error body */ }
      if (!r.ok) throw new Error(errorText(data, r.status));
      return data;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The request took too long. Please try again.");
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  // ---------------------------------------------------------------- symptom chips
  function renderChips() {
    const list = $("chips");
    list.replaceChildren(...state.symptoms.map((s) => {
      const li = el("li", "chip");
      li.append(el("span", "", s));
      const b = el("button", "", "×");
      b.type = "button";
      b.setAttribute("aria-label", "Remove " + s);
      b.addEventListener("click", () => { removeSymptom(s); $("symptom-input").focus(); });
      li.append(b);
      return li;
    }));
    document.querySelectorAll("#quick .pill").forEach((p) =>
      p.setAttribute("aria-pressed", String(state.symptoms.includes(p.dataset.symptom))));
  }

  function addSymptoms(raw) {
    let error = "";
    for (const p of String(raw).split(/[,;\n]+/).map(norm).filter(Boolean)) {
      if (p.length > MAX_LEN) { error = "“" + p.slice(0, 20) + "…” is too long (max " + MAX_LEN + " characters)."; continue; }
      if (state.symptoms.includes(p)) continue;
      if (state.symptoms.length >= MAX_SYMPTOMS) { error = "You can add up to " + MAX_SYMPTOMS + " symptoms."; break; }
      state.symptoms.push(p);
    }
    renderChips();
    setStatus(error, Boolean(error));
  }

  function removeSymptom(s) {
    state.symptoms = state.symptoms.filter((x) => x !== s);
    renderChips();
  }

  function setupInput() {
    const input = $("symptom-input");
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") {
        if (input.value.trim() || e.key === ",") { e.preventDefault(); addSymptoms(input.value); input.value = ""; }
      } else if (e.key === "Backspace" && !input.value && state.symptoms.length) {
        removeSymptom(state.symptoms[state.symptoms.length - 1]);
      }
    });
    // choosing a suggestion from the datalist fires an input event without typed text
    input.addEventListener("input", (e) => {
      if ((e.inputType === undefined || e.inputType === "insertReplacementText") && input.value.trim()) {
        addSymptoms(input.value); input.value = "";
      }
    });
    $("chipbox").addEventListener("click", () => input.focus());

    $("quick").replaceChildren(...QUICK.map((s) => {
      const b = el("button", "pill", s);
      b.type = "button"; b.dataset.symptom = s; b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => (state.symptoms.includes(s) ? removeSymptom(s) : addSymptoms(s)));
      return b;
    }));
    $("examples").replaceChildren(...EXAMPLES.map((ex) => {
      const b = el("button", "pill", ex.join(", "));
      b.type = "button";
      b.addEventListener("click", () => { state.symptoms = []; addSymptoms(ex.join(",")); });
      return b;
    }));

    api("/api/symptoms").then((d) => {
      $("symptom-list").replaceChildren(...d.symptoms.map((s) => { const o = el("option"); o.value = s; return o; }));
    }).catch(() => { /* suggestions are optional */ });
  }

  // ---------------------------------------------------------------- photos
  function renderThumbs() {
    $("thumbs").replaceChildren(...state.files.map((f) => {
      const li = el("li", "thumb");
      const img = el("img"); img.src = f.preview; img.alt = "Selected photo " + f.file.name;
      const b = el("button", "", "×"); b.type = "button"; b.setAttribute("aria-label", "Remove photo " + f.file.name);
      b.addEventListener("click", () => {
        URL.revokeObjectURL(f.preview);
        state.files = state.files.filter((x) => x !== f);
        renderThumbs();
      });
      li.append(img, b);
      return li;
    }));
  }

  function addFiles(fileList) {
    let error = "";
    for (const file of Array.from(fileList)) {
      if (!ALLOWED_TYPES.includes(file.type)) { error = "“" + file.name + "” is not a JPEG, PNG or WEBP image."; continue; }
      if (file.size > MAX_BYTES) { error = "“" + file.name + "” is larger than 5 MB."; continue; }
      if (state.files.some((x) => x.file.name === file.name && x.file.size === file.size)) continue;
      if (state.files.length >= MAX_IMAGES) { error = "You can add up to " + MAX_IMAGES + " photos."; break; }
      state.files.push({ file, preview: URL.createObjectURL(file) });
    }
    renderThumbs();
    setStatus(error, Boolean(error));
  }

  function setupDrop() {
    const drop = $("drop"), input = $("images");
    input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
  }

  // ---------------------------------------------------------------- results panel
  function showEmpty() {
    const box = el("div", "empty");
    box.append(el("div", "empty-icon", "🩺"), el("p", "empty-title", "Your results will appear here"));
    const sub = el("p", "empty-sub");
    sub.append("Add at least one symptom (three or more works best) and press ", el("strong", "", "Check symptoms"), ".");
    box.append(sub);
    $("results-body").replaceChildren(box);
  }

  function showLoading() {
    const wrap = el("div"); wrap.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) {
      const sk = el("div", "skeleton"); sk.style.marginBottom = "12px";
      sk.append(el("div", "sk l1"), el("div", "sk l2"), el("div", "sk l3"));
      wrap.append(sk);
    }
    $("results-body").replaceChildren(wrap);
  }

  function showError(message) {
    const a = el("div", "alert error", message); a.setAttribute("role", "alert");
    $("results-body").replaceChildren(a);
    setStatus("");
  }

  function makeExplainer(cond, btn, out) {
    const ctl = { done: false, running: false };
    ctl.run = async () => {
      if (ctl.done || ctl.running) return;
      ctl.running = true; btn.disabled = true; btn.textContent = "Explaining…";
      try {
        const data = await api("/api/explain", {
          method: "POST", headers: { "Content-Type": "application/json" }, timeout: 40000,
          body: JSON.stringify({ condition: cond.name, symptoms: state.matched }),
        });
        const label = data.source === "llm" ? (data.cached ? "AI explanation · cached" : "AI explanation") : "Basic description · AI unavailable";
        out.replaceChildren(el("p", "", data.explanation), el("span", "badge", label));
        out.hidden = false; btn.hidden = true; ctl.done = true;
      } catch (e) {
        out.replaceChildren(el("p", "error", e.message));
        out.hidden = false; btn.disabled = false; btn.textContent = "Try again";
      } finally {
        ctl.running = false;
      }
    };
    btn.addEventListener("click", ctl.run);
    return ctl;
  }

  function renderCondition(cond, index) {
    const li = el("li", "card" + (index === 0 ? " top-card" : ""));
    const head = el("div", "card-head");
    head.append(el("span", "rank", String(index + 1)), el("h3", "name", cond.name), el("span", "pct", pct(cond.probability)));
    const bar = el("div", "bar"); bar.setAttribute("role", "img"); bar.setAttribute("aria-label", pct(cond.probability) + " match");
    const fill = el("span"); fill.style.width = Math.max(2, Math.round(cond.probability * 100)) + "%"; bar.append(fill);
    li.append(head, bar);
    if (cond.description) li.append(el("p", "desc", cond.description));
    if (cond.precautions && cond.precautions.length) {
      const d = el("details", "prec"); d.append(el("summary", "", "General precautions"));
      const ul = el("ul"); cond.precautions.forEach((p) => ul.append(el("li", "", p))); d.append(ul); li.append(d);
    }
    const actions = el("div", "card-actions");
    const btn = el("button", "secondary", "Explain this condition"); btn.type = "button";
    const out = el("div", "explanation"); out.hidden = true;
    actions.append(btn); li.append(actions, out);
    state.cards.push(makeExplainer(cond, btn, out));
    return li;
  }

  function renderResults(data) {
    state.matched = data.matched_symptoms; state.cards = [];
    const body = $("results-body"); body.replaceChildren();

    if (data.urgent_warning) {
      const a = el("div", "alert urgent", "⚠ " + data.urgent_warning); a.setAttribute("role", "alert"); body.append(a);
    }
    (data.warnings || []).forEach((w) => body.append(el("div", "alert info", w)));

    const sum = el("div", "summary"); sum.append(el("p", "label-sm", "Based on"));
    const chips = el("ul", "chips");
    data.matched_symptoms.forEach((s) => chips.append(el("li", "chip readonly", s.replaceAll("_", " "))));
    sum.append(chips);
    if (data.image_description) sum.append(el("p", "", "From your photo(s): " + data.image_description));
    if (data.unmatched_symptoms.length) sum.append(el("p", "", "Not recognised: " + data.unmatched_symptoms.join(", ") + ". Try a more common name."));
    if (data.matched_symptoms.length < 2) sum.append(el("p", "", "Tip: results are more reliable with three or more symptoms."));
    body.append(sum);

    const bar = el("div", "toolbar");
    bar.append(el("span", "count", data.conditions.length + " possible condition" + (data.conditions.length === 1 ? "" : "s") + (data.cached ? " · from cache" : "")));
    const all = el("button", "secondary", "Explain all"); all.type = "button";
    all.addEventListener("click", async () => {
      all.disabled = true;
      for (const c of state.cards) { await c.run(); }   // one at a time, to be gentle on the LLM
      all.hidden = true;
    });
    bar.append(all); body.append(bar);

    const ul = el("ul", "cards");
    data.conditions.forEach((c, i) => ul.append(renderCondition(c, i)));
    body.append(ul, el("p", "disclaimer", data.disclaimer));
  }

  // ---------------------------------------------------------------- submit / reset
  function setBusy(b) {
    state.busy = b;
    const btn = $("submit-btn");
    btn.disabled = b; btn.textContent = b ? "Checking…" : "Check symptoms";
  }

  async function onSubmit(ev) {
    ev.preventDefault();
    if (state.busy) return;
    const input = $("symptom-input");
    addSymptoms(input.value); input.value = "";
    if (!state.symptoms.length && !state.files.length) { setStatus("Add at least one symptom or a photo.", true); input.focus(); return; }
    const text = state.symptoms.join(", ");
    if (text.length > MAX_TEXT) { setStatus("That is too much text. Please remove a few symptoms.", true); return; }

    setBusy(true); setStatus(""); showLoading();
    try {
      let data;
      if (state.files.length) {                     // photos present -> multipart image endpoint
        const fd = new FormData();
        fd.append("symptoms", text);
        state.files.forEach((f) => fd.append("images", f.file));
        data = await api("/api/predict/image", { method: "POST", body: fd, timeout: 60000 });
      } else {                                      // text only -> JSON endpoint
        data = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symptoms: text }) });
      }
      renderResults(data);
      const r = $("results"); r.focus({ preventScroll: true }); r.scrollIntoView({ block: "start" });
    } catch (e) {
      showError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function onReset() {
    state.symptoms = []; state.matched = []; state.cards = [];
    state.files.forEach((f) => URL.revokeObjectURL(f.preview)); state.files = [];
    $("symptom-input").value = "";
    renderChips(); renderThumbs(); showEmpty(); setStatus("");
    $("symptom-input").focus();
  }

  $("symptom-form").addEventListener("submit", onSubmit);
  $("reset-btn").addEventListener("click", onReset);
  setupInput(); setupDrop(); renderChips();
})();
