// hover_test.js - injected into a test copy of the page; writes each chart's tooltip text into <div id="HOVERTEST">.
window.addEventListener("load", () => setTimeout(() => {
  const out = document.createElement("div"); out.id = "HOVERTEST"; document.body.appendChild(out);
  const tip = document.getElementById("tip"), txt = () => tip.textContent.replace(/\s+/g, " ").trim();
  const at = (node, fx, fy, type) => { const r = node.getBoundingClientRect();
    node.dispatchEvent(new PointerEvent(type || "pointermove", {bubbles: true, clientX: r.left + r.width * fx, clientY: r.top + r.height * fy})); };
  const res = {};
  const probe = (name, fn) => { tip.textContent = ""; try { res[name] = fn(); } catch (e) { res[name] = "ERROR " + e.message; } };
  probe("chart", () => { at(document.querySelector("#chart rect[style*=crosshair]"), 0.5, 0.5); return txt(); });
  probe("slot", () => { const s = document.getElementById("slot"); if (s.style.display === "none") return "n/a";
    at(document.querySelector("#slot rect[style*=crosshair]"), 0.3, 0.5); return txt(); });
  const first = (sel, name) => probe(name, () => { const n = document.querySelector(sel); if (!n) return "ERROR no mark";
    at(n, 0.5, 0.5, "pointerenter"); return txt(); });
  first("#cards svg g[style*=help]", "card");
  first("#frontier g[style*=help]", "frontier");
  first("#calc-svg g[style*=help]", "calc");
  first("#fusion g[style*=help]", "fusion");
  first("#lat g[style*=help]", "latency");
  first("#m8 g[style*=help]", "more8");
  first("#m6 g[style*=help]", "more6");
  out.textContent = JSON.stringify(res);
}, 800));
