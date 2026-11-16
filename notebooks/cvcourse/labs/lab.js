/* Shared toolkit for the interactive labs. No dependencies, works offline.
 * Authoring guide: README.md in this folder. Everything hangs off the global `Lab`.
 */
window.Lab = (() => {
  "use strict";
  const L = {};

  // ---------------------------------------------------------------- shell
  // Any script error becomes a visible red banner (so a broken lab is obvious in class and in screenshots).
  function banner(msg) {
    let b = document.querySelector(".lab-error");
    if (!b) { b = document.createElement("div"); b.className = "lab-error"; (document.body || document.documentElement).appendChild(b); }
    b.textContent += (b.textContent ? "\n" : "") + "Lab error: " + msg;
  }
  window.addEventListener("error", (e) => {
    if (/ResizeObserver loop/.test(e.message || "")) return;   // a browser notice about layout timing, not a script error
    banner(`${e.message} (line ${e.lineno})`);
  });
  window.addEventListener("unhandledrejection", (e) => banner(String(e.reason)));

  const embedded = window.self !== window.top;
  function fitFrame() {
    // Inside a notebook the lab lives in a same-origin srcdoc iframe: grow it to fit the content.
    try {
      const f = window.frameElement;
      if (!f || f.closest(".lab-embed")) return;   // a lecture slide gives the lab a fixed box: scroll inside it
      const h = Math.ceil(document.body.offsetHeight) + 2 + "px";
      if (f.style.height !== h) f.style.height = h;
    } catch (_) { /* cross-origin host: keep the height given by labs.show() */ }
  }
  // Theme: #theme=light|dark (or labs.show(name, theme="dark")) wins; otherwise follow the notebook host
  // (JupyterLab / VS Code) when we can see it, else the operating system setting (handled in lab.css).
  (function pickTheme() {
    let t = (window.LAB_STATE || {}).theme || new URLSearchParams(location.hash.replace(/^#/, "")).get("theme");
    if (!t && embedded) {
      try {
        const b = window.parent.document.body;
        if (b.dataset.jpThemeLight) t = b.dataset.jpThemeLight === "true" ? "light" : "dark";
        else if (b.classList.contains("vscode-dark") || b.classList.contains("vscode-high-contrast")) t = "dark";
        else if (b.classList.contains("vscode-light")) t = "light";
      } catch (_) { /* cross-origin host */ }
    }
    if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  })();
  document.addEventListener("DOMContentLoaded", () => {
    if (!embedded) document.body.classList.add("standalone");
    if (embedded && window.ResizeObserver) new ResizeObserver(fitFrame).observe(document.body);
    fitFrame();
  });

  /** Initial state = defaults, overridden by labs.show(name, key=value) and by the URL hash (#key=value&k2=v2). */
  L.params = (defaults = {}) => {
    const out = { ...defaults };
    const apply = (k, v) => {
      if (!(k in defaults)) return;
      const d = defaults[k];
      if (typeof d === "number") { const n = Number(v); if (!Number.isNaN(n)) out[k] = n; }
      else if (typeof d === "boolean") out[k] = v === true || v === "true" || v === "1" || v === 1;
      else out[k] = String(v);
    };
    for (const [k, v] of Object.entries(window.LAB_STATE || {})) apply(k, v);
    for (const [k, v] of new URLSearchParams(location.hash.replace(/^#/, ""))) apply(k, v);
    return out;
  };

  // ---------------------------------------------------------------- small utils
  L.clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  L.lerp = (a, b, t) => a + (b - a) * t;
  L.fmt = (x, digits = 2) => (Number.isFinite(x) ? x.toFixed(digits) : "-");
  /** Deterministic random numbers in [0, 1): const rnd = Lab.rng(1); rnd(). */
  L.rng = (seed = 1) => {
    let s = seed >>> 0 || 1;
    return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  };
  /** Lab.el("div", {class: "x", onclick: f, text: "hi"}, child1, child2) */
  L.el = (tag, props = {}, ...children) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null) e.append(c);
    return e;
  };
  L.$ = (sel) => document.querySelector(sel);
  const host = (p) => (typeof p === "string" ? document.querySelector(p) : p);

  /** Coalesce redraws: const redraw = Lab.loop(draw); call redraw() from every handler. Also redraws on theme change. */
  L.loop = (draw) => {
    let queued = false;
    const run = () => { queued = false; draw(); };
    const run2 = () => { if (queued) run(); };
    // rAF does not fire in hidden tabs or some headless runs; the timer is the fallback (whichever comes first draws).
    const schedule = () => { if (!queued) { queued = true; requestAnimationFrame(run2); setTimeout(run2, 60); } };
    themeListeners.push(schedule);
    schedule();
    return schedule;
  };

  // ---------------------------------------------------------------- colour
  let cssCache = {};
  const themeListeners = [];
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { cssCache = {}; themeListeners.forEach((f) => f()); });
  }
  /** Value of a CSS variable from lab.css, e.g. Lab.color("--c1"). Use roles, not hex, so dark mode works. */
  L.color = (name) => cssCache[name] || (cssCache[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim());
  /** "#rrggbb" -> [r, g, b] */
  L.rgb = (hex) => { const h = hex.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
  const mix3 = (a, b, t) => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t));
  const css3 = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  /** Diverging scale for signed values: v in [-1, 1]; negative = red, 0 = neutral, positive = blue. Returns [r,g,b]. */
  L.divergingRGB = (v) => {
    v = L.clamp(v, -1, 1);
    const mid = L.rgb(L.color("--div-mid"));
    return v >= 0 ? mix3(mid, L.rgb(L.color("--div-pos")), v) : mix3(mid, L.rgb(L.color("--div-neg")), -v);
  };
  /** Sequential scale for magnitudes: t in [0, 1]. Returns [r,g,b]. */
  L.sequentialRGB = (t) => mix3(L.rgb(L.color("--seq-lo")), L.rgb(L.color("--seq-hi")), L.clamp(t, 0, 1));
  L.grayRGB = (t) => { const g = Math.round(255 * L.clamp(t, 0, 1)); return [g, g, g]; };
  L.diverging = (v) => css3(L.divergingRGB(v));
  L.sequential = (t) => css3(L.sequentialRGB(t));
  /** Readable text colour on top of an [r,g,b] fill. */
  L.inkOn = (c) => ((0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) > 150 ? "#0b0b0b" : "#ffffff");
  /** Same colour with alpha: Lab.alpha("--c1", 0.25) */
  L.alpha = (nameOrHex, a) => { const c = L.rgb(nameOrHex.startsWith("--") ? L.color(nameOrHex) : nameOrHex); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };

  // ---------------------------------------------------------------- controls
  function ctl(parent, label) {
    const value = L.el("span", { class: "ctl-value" });
    const head = L.el("span", { class: "ctl-label" }, L.el("span", { text: label || "" }), value);
    const wrap = L.el("label", { class: "ctl" }, head);
    host(parent).append(wrap);
    return { wrap, value };
  }
  /** Lab.slider(parent, {label, min, max, step, value, fmt: v => text, onInput: v => {}}) -> {value, set(v), el} */
  L.slider = (parent, o) => {
    const { wrap, value } = ctl(parent, o.label);
    const input = L.el("input", { type: "range", min: o.min, max: o.max, step: o.step ?? 1, "aria-label": o.label });
    input.value = o.value;
    const fmt = o.fmt || ((v) => String(v));
    const show = () => { value.textContent = fmt(Number(input.value)); };
    input.addEventListener("input", () => { show(); o.onInput && o.onInput(Number(input.value)); });
    wrap.append(input); show();
    return { el: wrap, input, get value() { return Number(input.value); }, set(v) { input.value = v; show(); } };
  };
  const normOpts = (opts) => opts.map((x) => (typeof x === "object" ? x : { value: x, label: String(x) }));
  /** Lab.select(parent, {label, options: ["a", {value, label}], value, onChange}) */
  L.select = (parent, o) => {
    const { wrap } = ctl(parent, o.label);
    const sel = L.el("select", { "aria-label": o.label }, normOpts(o.options).map((x) => L.el("option", { value: x.value, text: x.label })));
    sel.value = o.value;
    sel.addEventListener("change", () => o.onChange && o.onChange(sel.value));
    wrap.append(sel);
    return { el: wrap, get value() { return sel.value; }, set(v) { sel.value = v; } };
  };
  /** Segmented buttons for 2-5 exclusive choices. Values keep their type. */
  L.segmented = (parent, o) => {
    const opts = normOpts(o.options);
    let current = o.value;
    const seg = L.el("div", { class: "seg", role: "group", "aria-label": o.label });
    const btns = opts.map((x) => {
      const b = L.el("button", { type: "button", text: x.label });
      b.addEventListener("click", () => { api.set(x.value); o.onChange && o.onChange(x.value); });
      seg.append(b); return b;
    });
    const paint = () => btns.forEach((b, i) => b.setAttribute("aria-pressed", String(opts[i].value === current)));
    const wrap = L.el("div", { class: "ctl" }, o.label ? L.el("span", { class: "ctl-label", text: o.label }) : null, seg);
    host(parent).append(wrap);
    const api = { el: wrap, get value() { return current; }, set(v) { current = v; paint(); } };
    paint();
    return api;
  };
  /** Checkbox. Lab.toggle(parent, {label, value, onChange}) */
  L.toggle = (parent, o) => {
    const input = L.el("input", { type: "checkbox" });
    input.checked = !!o.value;
    input.addEventListener("change", () => o.onChange && o.onChange(input.checked));
    const wrap = L.el("label", { class: "ctl ctl-check" }, input, L.el("span", { text: o.label }));
    host(parent).append(wrap);
    return { el: wrap, get value() { return input.checked; }, set(v) { input.checked = !!v; } };
  };
  L.button = (parent, text, onClick, cls = "") => {
    const b = L.el("button", { type: "button", class: "btn " + cls, text });
    b.addEventListener("click", onClick);
    host(parent).append(b);
    return b;
  };
  /** A live number tile. const r = Lab.readout(parent, "IoU"); r.set("0.53"); r.set("0.71", "good" | "bad" | "") */
  L.readout = (parent, label) => {
    const v = L.el("span", { class: "v", text: "-" });
    const box = L.el("div", { class: "readout" }, L.el("span", { class: "k", text: label }), v);
    host(parent).append(box);
    return { el: box, set(text, tone = "") { v.textContent = text; box.dataset.tone = tone; } };
  };

  // ---------------------------------------------------------------- canvas
  /** Hi-DPI canvas with a logical coordinate system of w x h units; shrinks responsively.
   *  const c = Lab.canvas(parent, 420, 300); c.ctx.fillRect(...); c.clear(); */
  L.canvas = (parent, w, h, { framed = false } = {}) => {
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const canvas = L.el("canvas", { class: "lab-canvas" + (framed ? " framed" : ""), width: Math.round(w * dpr), height: Math.round(h * dpr) });
    canvas.style.width = w + "px";
    canvas.style.aspectRatio = `${w} / ${h}`;
    host(parent).append(canvas);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { canvas, ctx, w, h, clear() { ctx.clearRect(0, 0, w, h); } };
  };
  /** Pointer events in logical canvas units. handlers: down(p), move(p) while dragging, up(p), hover(p), leave().
   *  p = {x, y}. Works with mouse, pen and touch. */
  L.pointer = (c, handlers) => {
    const cv = c.canvas;
    const pos = (e) => { const r = cv.getBoundingClientRect(); return { x: ((e.clientX - r.left) / r.width) * c.w, y: ((e.clientY - r.top) / r.height) * c.h }; };
    let dragging = false;
    cv.addEventListener("pointerdown", (e) => { dragging = true; cv.setPointerCapture(e.pointerId); handlers.down && handlers.down(pos(e), e); e.preventDefault(); });
    cv.addEventListener("pointermove", (e) => { const p = pos(e); if (dragging) handlers.move && handlers.move(p, e); else handlers.hover && handlers.hover(p, e); });
    const end = (e) => { if (dragging) { dragging = false; handlers.up && handlers.up(pos(e), e); } };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
    cv.addEventListener("pointerleave", () => { if (!dragging) handlers.leave && handlers.leave(); });
  };
  /** Text with sane defaults. opts: {color, size, weight, align, baseline, mono} */
  L.text = (ctx, str, x, y, o = {}) => {
    ctx.save();
    ctx.font = `${o.weight || 400} ${o.size || 12}px ${o.mono ? 'ui-monospace, Consolas, monospace' : 'system-ui, -apple-system, "Segoe UI", sans-serif'}`;
    ctx.fillStyle = o.color || L.color("--ink-2");
    ctx.textAlign = o.align || "left";
    ctx.textBaseline = o.baseline || "alphabetic";
    ctx.fillText(str, x, y);
    ctx.restore();
  };
  /** Paint a scalar field (row-major array, cols x rows) as a pixelated image into the rect (x, y, w, h).
   *  map(value, index) -> [r, g, b]. Use for photos and feature maps. */
  L.image = (ctx, data, cols, rows, x, y, w, h, map = (v) => L.grayRGB(v)) => {
    const off = document.createElement("canvas"); off.width = cols; off.height = rows;
    const octx = off.getContext("2d"); const img = octx.createImageData(cols, rows);
    for (let i = 0; i < cols * rows; i++) { const c = map(data[i], i); img.data[4 * i] = c[0]; img.data[4 * i + 1] = c[1]; img.data[4 * i + 2] = c[2]; img.data[4 * i + 3] = 255; }
    octx.putImageData(img, 0, 0);
    ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, x, y, w, h); ctx.restore();
  };
  /** Paint a small matrix as rounded cells with the numbers written in them.
   *  o: {color: v => [r,g,b], fmt: v => text, gap, size (font px), values (default true), highlight: (i, j) => bool} */
  L.cells = (ctx, data, cols, rows, x, y, w, h, o = {}) => {
    const gap = o.gap ?? 2, cw = w / cols, ch = h / rows;
    const color = o.color || ((v) => L.divergingRGB(v));
    const fmt = o.fmt || ((v) => (Number.isInteger(v) ? String(v) : v.toFixed(1)));
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = data[i * cols + j];
      if (v == null || Number.isNaN(v)) { ctx.fillStyle = L.color("--wash"); ctx.fillRect(x + j * cw + gap / 2, y + i * ch + gap / 2, cw - gap, ch - gap); continue; }
      const c = color(v, i, j);
      ctx.fillStyle = css3(c);
      ctx.beginPath(); ctx.roundRect(x + j * cw + gap / 2, y + i * ch + gap / 2, cw - gap, ch - gap, Math.min(4, cw / 6)); ctx.fill();
      if (o.values !== false && cw >= 16) L.text(ctx, fmt(v), x + (j + 0.5) * cw, y + (i + 0.5) * ch + 0.5, { color: L.inkOn(c), align: "center", baseline: "middle", size: o.size || Math.min(13, cw * 0.36) });
      if (o.highlight && o.highlight(i, j)) { ctx.strokeStyle = L.color("--hl"); ctx.lineWidth = 2.5; ctx.strokeRect(x + j * cw + 1, y + i * ch + 1, cw - 2, ch - 2); }
    }
  };
  /** Axes for a line / scatter chart. Draws grid, ticks and labels, returns the scales.
   *  const ax = Lab.axes(ctx, {x, y, w, h, xmin, xmax, ymin, ymax, xlabel, ylabel, xticks: [..], yticks: [..], xfmt, yfmt});
   *  ax.sx(v), ax.sy(v) data -> px;  ax.ix(px), ax.iy(px) px -> data;  ax.line(points, color, width, dash); ax.dot(x, y, color, r) */
  L.axes = (ctx, o) => {
    const { x, y, w, h, xmin, xmax, ymin, ymax } = o;
    const sx = (v) => x + ((v - xmin) / (xmax - xmin)) * w;
    const sy = (v) => y + h - ((v - ymin) / (ymax - ymin)) * h;
    const ticks = (lo, hi, n = 5) => Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
    const xt = o.xticks || ticks(xmin, xmax), yt = o.yticks || ticks(ymin, ymax);
    const nice = (v) => (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v)) : String(+v.toFixed(2)));
    ctx.save();
    ctx.lineWidth = 1; ctx.strokeStyle = L.color("--grid");
    for (const t of yt) { ctx.beginPath(); ctx.moveTo(x, Math.round(sy(t)) + 0.5); ctx.lineTo(x + w, Math.round(sy(t)) + 0.5); ctx.stroke(); }
    ctx.strokeStyle = L.color("--axis");
    ctx.beginPath(); ctx.moveTo(x, y + h + 0.5); ctx.lineTo(x + w, y + h + 0.5); ctx.stroke();
    for (const t of xt) L.text(ctx, (o.xfmt || nice)(t), sx(t), y + h + 14, { align: "center", color: L.color("--muted"), size: 11 });
    for (const t of yt) L.text(ctx, (o.yfmt || nice)(t), x - 6, sy(t), { align: "right", baseline: "middle", color: L.color("--muted"), size: 11 });
    if (o.xlabel) L.text(ctx, o.xlabel, x + w / 2, y + h + 30, { align: "center", size: 12 });
    if (o.ylabel) { ctx.save(); ctx.translate(x - 38, y + h / 2); ctx.rotate(-Math.PI / 2); L.text(ctx, o.ylabel, 0, 0, { align: "center", size: 12 }); ctx.restore(); }
    ctx.restore();
    const ax = {
      sx, sy,
      ix: (px) => xmin + ((px - x) / w) * (xmax - xmin),
      iy: (py) => ymin + ((y + h - py) / h) * (ymax - ymin),
      /** points: [[x, y], ...] in data units; clipped to the plot area */
      line(points, color, width = 2, dash = []) {
        ctx.save(); ctx.beginPath(); ctx.rect(x, y - 1, w, h + 2); ctx.clip();
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.setLineDash(dash);
        ctx.beginPath(); points.forEach((p, i) => (i ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1])))); ctx.stroke();
        ctx.restore();
      },
      dot(px, py, color, r = 4.5) {
        ctx.save(); ctx.fillStyle = color; ctx.strokeStyle = L.color("--surface"); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(sx(px), sy(py), r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); ctx.restore();
      },
    };
    return ax;
  };
  /** Box outline with an optional label chip. box = [x1, y1, x2, y2] in canvas units. o: {color, width, dash, fill, label} */
  L.box = (ctx, b, o = {}) => {
    const color = o.color || L.color("--c1");
    ctx.save();
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); }
    ctx.strokeStyle = color; ctx.lineWidth = o.width || 2; ctx.setLineDash(o.dash || []);
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    if (o.label) {
      ctx.font = '600 11px system-ui, -apple-system, "Segoe UI", sans-serif';
      const tw = ctx.measureText(o.label).width + 8;
      const ty = b[1] - 16 < 0 ? b[1] : b[1] - 16;
      ctx.fillStyle = color; ctx.fillRect(b[0] - 1, ty, tw, 16);
      ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(o.label, b[0] + 3, ty + 8.5);
    }
    ctx.restore();
  };

  // ---------------------------------------------------------------- sample image
  const PHOTO = "data:image/jpeg;base64,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";
  /** The course's sample photo (Grace Hopper, as in the notebooks), 128 x 128.
   *  Lab.photo().then(({w, h, gray, rgb, img}) => ...)  gray: Float32Array in [0,1] (w*h); rgb: Float32Array (w*h*3); img: HTMLImageElement */
  L.photo = () => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = img.width, h = img.height;
      const off = document.createElement("canvas"); off.width = w; off.height = h;
      const octx = off.getContext("2d"); octx.drawImage(img, 0, 0);
      const d = octx.getImageData(0, 0, w, h).data;
      const gray = new Float32Array(w * h), rgb = new Float32Array(w * h * 3);
      for (let i = 0; i < w * h; i++) {
        const r = d[4 * i] / 255, g = d[4 * i + 1] / 255, b = d[4 * i + 2] / 255;
        rgb[3 * i] = r; rgb[3 * i + 1] = g; rgb[3 * i + 2] = b;
        gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      }
      resolve({ w, h, gray, rgb, img });
    };
    img.onerror = () => reject(new Error("sample photo failed to decode"));
    img.src = PHOTO;
  });

  return L;
})();
