// build_page.js - assemble docs/index.html, a single self-contained page that needs no network access to view.
//   python site/build_data.py build/data.json
//   node site/build_page.js site/template.html build/data.json docs/index.html
// - every \( \) and \[ \] in the template is typeset with KaTeX here, at build time;
// - every expression the page's script shows is pre-rendered into window.TEX (a miss shows as <html data-tex-miss>);
// - KaTeX's stylesheet and fonts, Inter and IBM Plex Mono are embedded as data URIs (licences in site/vendor/).
const fs = require("fs"), path = require("path");
const [tplP, dataP, outP] = process.argv.slice(2);
if (!outP) throw new Error("usage: node site/build_page.js TEMPLATE DATA.json OUT.html");
const V = path.join(__dirname, "vendor");
const katex = require(path.join(V, "katex", "katex.min.js"));
const b64 = f => fs.readFileSync(f).toString("base64");
// receiver colours, one per receiver across the page: classical, network alone, bounded output; allowed range
const SLOT = {c: "#7c2d12", n: "#059669", o: "#e00800", band: "#e00800"};

let t = fs.readFileSync(tplP, "utf8");
const data = fs.readFileSync(dataP, "utf8"); JSON.parse(data);
const render = (tex, display) => katex.renderToString(tex, {displayMode: display, throwOnError: true, strict: "ignore"});

// the review banner of the template is not part of the published page
t = t.replace(/<div class="mock">[\s\S]*?<\/div>\n/, "");
if (t.includes('class="mock"')) throw new Error("the review banner was not removed");

// static mathematics in the markup (everything before the tooltip element; the script holds none)
const cut = t.indexOf('<div class="tip" id="tip"');
if (cut < 0) throw new Error("tooltip anchor not found");
let head = t.slice(0, cut), tail = t.slice(cut), nStatic = 0;
head = head.replace(/\\\[([\s\S]+?)\\\]/g, (m, x) => { nStatic++; return render(x, true); })
           .replace(/\\\(([\s\S]+?)\\\)/g, (m, x) => { nStatic++; return render(x, false); });
if (/\\\(|\\\[/.test(head)) throw new Error("an unrendered math delimiter is left in the markup");

// expressions the script typesets at view time
const LT = ["0", "\\ln 2", "\\ln 4", "\\ln 8", "\\ln 16", "\\infty"];
const dyn = new Set(["\\Delta_{\\max}", "E_{\\mathrm{b}}/N_0", "\\Delta_{\\max}=0", "\\Delta_{\\max}\\to\\infty",
  "L_{\\mathrm{c}}", "L_{\\mathrm{R3}}", "L_{\\mathrm{R5}}", "L_{\\mathrm{R3}}-L_{\\mathrm{c}}", "L_{\\mathrm{c}}\\pm\\Delta_{\\max}"]);
LT.forEach(x => { dyn.add(x); dyn.add("\\Delta_{\\max}=" + x); dyn.add("\\le " + x); });
["\\lambda=0", "\\lambda=0.25", "\\lambda=0.5", "\\lambda=0.75", "\\lambda=1", "\\tau", "\\tau=0", "\\tau=\\infty", "g", "g=0",
 "g=\\infty", "C=\\ln 2", "C=\\ln 4", "C=\\ln 8", "C=\\ln 16", "C\\to\\infty", "P(b=1)", "e^{\\Delta_{\\max}}",
 "\\lvert L_{\\mathrm{c}}\\rvert"].forEach(x => dyn.add(x));     // fusion-rule settings and the one-bit calculator
const TEX = {}; dyn.forEach(x => { TEX[x] = render(x, false); });
const texJson = JSON.stringify(TEX).replace(/<\//g, "<\\/");

// embedded stylesheet: KaTeX with its woff2 fonts inline, then the two text faces (latin subsets)
let kcss = fs.readFileSync(path.join(V, "katex", "katex.min.css"), "utf8"), nFonts = 0;
kcss = kcss.replace(/src:url\(fonts\/(KaTeX_[\w-]+)\.woff2\) format\("woff2"\)[^;}]*/g, (m, name) => {
  nFonts++; return `src:url(data:font/woff2;base64,${b64(path.join(V, "katex", "fonts", name + ".woff2"))}) format("woff2")`;
});
if (/url\(fonts\//.test(kcss)) throw new Error("a KaTeX font reference was not inlined");
const faces = fs.readFileSync(path.join(V, "fonts", "latin_faces.tsv"), "utf8").trim().split("\n").map(l => l.split("\t"));
const inter = faces.find(f => f[0] === "Inter"), plex = faces.filter(f => f[0] === "IBM Plex Mono");
const file = f => path.join(V, "fonts", f[0].replace(/ /g, "") + "-" + path.basename(f[2]));
let fcss = `@font-face{font-family:"Inter";font-style:normal;font-weight:100 900;font-display:swap;src:url(data:font/woff2;base64,${b64(file(inter))}) format("woff2");unicode-range:${inter[3]}}`;
plex.forEach(f => { fcss += `@font-face{font-family:"IBM Plex Mono";font-style:normal;font-weight:${f[1]};font-display:swap;src:url(data:font/woff2;base64,${b64(file(f))}) format("woff2");unicode-range:${f[3]}}`; });

let out = (head + tail).replace("/*__ASSETS_CSS__*/", () => kcss + fcss).replace("__TEX__", () => texJson)
  .replace("__SLOT_PALETTE__", () => JSON.stringify(SLOT)).replace("__DATA__", () => data);
if (/https?:\/\/(fonts|cdn)\./.test(out)) throw new Error("an external asset reference is left");
if (/__(DATA|TEX|ASSETS_CSS|SLOT_PALETTE|BUILD)__/.test(out)) throw new Error("a placeholder is left");
fs.writeFileSync(outP, out);
console.log(`built ${outP}: ${out.length.toLocaleString("en")} bytes; static expressions ${nStatic}; view-time expressions ${dyn.size}; KaTeX fonts inlined ${nFonts}`);
