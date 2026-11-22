"""check_page.py PAGE - open the page in headless Chrome with all host resolution blocked and check, in four states:
typeset maths (no raw TeX, no plain 'Δmax', no TeX misses) and a working hover tooltip on each of the three graphs.
Exit 1 on any failure."""
import html as html_mod, json, os, re, shutil, subprocess, sys, tempfile
page = os.path.abspath(sys.argv[1]); here = os.path.dirname(os.path.abspath(__file__))
probe = open(os.path.join(here, "hover_test.js"), encoding="utf-8").read()
tmp = tempfile.mkdtemp(); test = os.path.join(tmp, "test.html")
html = open(page, encoding="utf-8").read()
assert html.count("</body>") == 1
open(test, "w", encoding="utf-8").write(html.replace("</body>", "<script>" + probe + "</script></body>"))
bad = 0
for st in ("c=cdlC_irr25&d=ln4", "c=irr20&d=unbounded", "c=joint&d=0", "c=delay300&d=ln8"):
    prof = os.path.join(tmp, "prof"); shutil.rmtree(prof, ignore_errors=True)
    dom = subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", f"--user-data-dir={prof}",
                          "--virtual-time-budget=6000", "--host-resolver-rules=MAP * ~NOTFOUND", "--dump-dom",
                          f"file://{test}#{st}"], capture_output=True, text=True).stdout
    miss = re.search(r'<html[^>]*data-tex-miss="([^"]*)"', dom)
    body = re.sub(r"<(script|style|annotation)\b.*?</\1>", "", re.sub(r'<div id="HOVERTEST">.*?</div>', "", dom, flags=re.S), flags=re.S)
    txt = re.sub(r"<[^>]+>", " ", body)
    raw = re.findall(r"\\[A-Za-z(\[]+", txt)
    hv = re.search(r'<div id="HOVERTEST">(\{.*?\})</div>', dom, re.S)
    hov = json.loads(html_mod.unescape(hv.group(1))) if hv else {}
    keys = ("chart", "slot", "card", "frontier", "calc", "fusion", "latency", "more8", "more6")
    ok_h = {k: bool(hov.get(k)) and not hov[k].startswith("ERROR") for k in keys}
    ok = not miss and not raw and "Δmax" not in txt and all(ok_h.values())
    bad += not ok
    nk = dom.count('class="katex"'); mtxt = miss.group(1) if miss else 0
    hs = " ".join(f"{k}={'ok' if ok_h[k] else 'NO'}" + ("(n/a)" if hov.get(k) == "n/a" else "") for k in keys); verdict = "PASS" if ok else "FAIL"
    print(f"{verdict} {st:<22} katex {nk:>3}  misses {mtxt}  raw TeX {len(raw)}  hover {hs}")
    if not all(ok_h.values()): print("   ", {k: (hov.get(k) or "")[:90] for k in ok_h})
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if bad else 0)
