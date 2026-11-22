// hover_test.js - injected into a test copy of the page; writes each graph's tooltip text into <div id="HOVERTEST">.
window.addEventListener("load", () => setTimeout(() => {
  const out = document.createElement("div"); out.id = "HOVERTEST"; document.body.appendChild(out);
  const tip = document.getElementById("tip"), txt = () => tip.textContent.replace(/\s+/g, " ").trim();
  const at = (node, fx, fy, type) => { const r = node.getBoundingClientRect();
    node.dispatchEvent(new PointerEvent(type || "pointermove", {bubbles: true, clientX: r.left + r.width * fx, clientY: r.top + r.height * fy})); };
  const res = {};
  try { at(document.querySelector("#chart rect[style*=crosshair]"), 0.5, 0.5); res.chart = txt(); } catch (e) { res.chart = "ERROR " + e.message; }
  tip.textContent = "";
  try { at(document.querySelector("#slot rect[style*=crosshair]"), 0.3, 0.5); res.slot = txt(); } catch (e) { res.slot = "ERROR " + e.message; }
  tip.textContent = "";
  try { at(document.querySelector("#cards svg g[style*=help]"), 0.5, 0.5, "pointerenter"); res.card = txt(); } catch (e) { res.card = "ERROR " + e.message; }
  out.textContent = JSON.stringify(res);
}, 800));
