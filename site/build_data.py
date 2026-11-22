"""build_data.py OUT.json - the data the page shows, built from results/ in this repository.

Checked against tests/reference_values.json (the published paper's tables, as printed) before anything is written:
  - every 10% error point and its cross-seed spread, for all seven conditions and six bounds (Table V);
  - the headline gain and the shares of it kept at ln 4, ln 8 and ln 16;
  - the fusion-rule operating points the paper prints (Table VII).
The latency and extra-experiment tables (II, IV-a, IV-b, VI, VIII) are taken from the paper as printed. Any mismatch stops
the build and nothing is written.

    python site/build_data.py build/data.json
"""
import csv, json, math, os, re, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DM = ["0", "ln2", "ln4", "ln8", "ln16", "unbounded"]
SEEDS = (42, 43, 44)
SHORT = {"cdlC_irr25": "Training conditions", "cdlD": "New channel: CDL-D", "cdlE": "New channel: CDL-E",
         "irr20": "Harsher IQ imbalance", "irr35": "Milder IQ imbalance", "joint": "Both shifts at once",
         "delay300": "Longer delay spread"}
FUSION_CONDS = ["cdlC_irr25", "irr20", "irr35", "cdlD", "cdlE"]


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def cell(s):
    """A printed table cell as data: '20.49 (0.06)' -> mean and SD; 'not reached*' / 'unreachable' -> not reached."""
    s = s.strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?: dB)? \((\d+(?:\.\d+)?)\)", s)
    if m:
        return {"text": s, "mean": float(m.group(1)), "sd": float(m.group(2))}
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?) ?(dB|ms|×)?", s)
    if m:
        return {"text": s, "mean": float(m.group(1)), "unit": m.group(2) or ""}
    if re.match(r"(not reached|unreachable)", s):
        return {"text": s, "nr": True}
    return {"text": s}


def cond_key(label):
    for pat, key in [(r"CDL-C, IRR 25", "cdlC_irr25"), (r"CDL-C, IRR 20", "irr20"), (r"CDL-C, IRR 35", "irr35"),
                     (r"CDL-D, IRR 20", "joint"), (r"CDL-D", "cdlD"), (r"CDL-E", "cdlE"), (r"CDL-C, 300 ns", "delay300")]:
        if label.startswith(pat.replace("\\", "")) or re.match(pat, label):
            return key
    return None


cfg = json.load(open(os.path.join(RES, "run_config.json")))
slots = json.load(open(os.path.join(RES, "llr_slots.json")))
ref = json.load(open(os.path.join(ROOT, "tests", "reference_values.json")))
if ref["dmax"] != DM:
    stop("the reference values use a different bound order")
n_checked = 0

# ---- the seven conditions: curves (mean, range, each seed) and 10% points, checked against Table V
data = {"conditions": [], "dmax": DM}
for key, short in SHORT.items():
    cc = cfg["conditions"][key]
    irr = int(cc["irr_db"]) if float(cc["irr_db"]).is_integer() else cc["irr_db"]
    detail = f"CDL-{cc['channel']} channel, IRR {irr} dB" + (f", {cc['delay_spread_ns']:g} ns delay spread" if cc.get("delay_spread_ns") else "")
    entry = {"key": key, "short": short, "detail": detail, "curves": {}, "crossing": {}}
    for d in DM:
        per = []
        for s in SEEDS:
            rows = list(csv.DictReader(open(os.path.join(RES, key, f"seed{s}", f"r5inf_bler_dmax{d}_irr{irr}.csv"))))
            per.append(([float(r["snr_db"]) for r in rows], [float(r["bler_mean"]) for r in rows]))
        snr = per[0][0]
        if any(p[0] != snr for p in per):
            stop(f"{key} {d}: SNR grids differ across seeds")
        cols = list(zip(*[p[1] for p in per]))
        entry["snr"] = snr
        entry["curves"][d] = {"mean": [round(sum(c) / len(c), 6) for c in cols], "lo": [round(min(c), 6) for c in cols],
                              "hi": [round(max(c), 6) for c in cols], "seeds": [[round(v, 6) for v in p[1]] for p in per]}
        xs = []
        for s in SEEDS:
            summ = {r["delta_max_name"]: r["snr10_db"] for r in csv.DictReader(open(os.path.join(RES, key, f"seed{s}", "r5inf_dmax_sweep_summary.csv")))}
            xs.append(None if summ[d] == "unreachable" else float(summ[d]))
        if all(v is not None for v in xs):
            m, sd = st.mean(xs), st.stdev(xs)
            entry["crossing"][d] = {"mean": round(m, 4), "sd": round(sd, 4), "seeds": xs}
            printed = f"{m:.2f} ({sd:.2f})"
        else:
            entry["crossing"][d] = {"mean": None, "floor": round(min(entry["curves"][d]["mean"]), 4), "seeds": xs}
            printed = "not reached"
        if not ref["table_v"][key][DM.index(d)].startswith(printed):
            stop(f"{key} {d}: computed '{printed}' but the paper prints '{ref['table_v'][key][DM.index(d)]}'")
        n_checked += 1
    entry["slot"] = slots.get(key)
    data["conditions"].append(entry)

c = data["conditions"][0]["crossing"]
gain = c["0"]["mean"] - c["unbounded"]["mean"]
shares = [round(100 * (c["0"]["mean"] - c[d]["mean"]) / gain) for d in ("ln4", "ln8", "ln16")]
if f"{gain:.2f}" != ref["headline_gain_db"] or shares != ref["shares_pct"]:
    stop(f"headline {gain:.2f} dB / {shares} differs from the paper's {ref['headline_gain_db']} dB / {ref['shares_pct']}")

# ---- fusion rules: per-seed operating points and gains over the classical receiver, checked against Table VII
rows = []
for s in SEEDS:
    rows += list(csv.DictReader(open(os.path.join(RES, "fusion", f"seed{s}", "a3_summary.csv"))))
fus = {}
for cond in FUSION_CONDS:
    pts = {}
    specs = sorted({(r["kind"], r["param"]) for r in rows if r["condition"] == cond})
    base = {int(r["seed"]): (None if r["crossing_db"] == "unreachable" else float(r["crossing_db"]))
            for r in rows if r["condition"] == cond and r["kind"] == "R5" and r["param"] == "0"}
    for kind, param in specs:
        sel = {int(r["seed"]): r for r in rows if r["condition"] == cond and r["kind"] == kind and r["param"] == param}
        if sorted(sel) != list(SEEDS):
            stop(f"fusion {cond} {kind}:{param}: not all three seeds present")
        xs = [None if sel[s]["crossing_db"] == "unreachable" else float(sel[s]["crossing_db"]) for s in SEEDS]
        p = {"seeds": xs, "min_bler": round(sum(float(sel[s]["min_bler"]) for s in SEEDS) / 3, 6)}
        if all(v is not None for v in xs):
            g = [base[s] - x for s, x in zip(SEEDS, xs)]
            p.update(mean=round(st.mean(xs), 4), sd=round(st.stdev(xs), 4), gain=round(st.mean(g), 4),
                     gain_lo=round(min(g), 4), gain_hi=round(max(g), 4))
        pts[f"{kind}:{param}"] = p
    fus[cond] = pts
t7 = ref["tables"]["VII"]
colmap = {}
for j, h in enumerate(t7[0]):
    if h.startswith("L2-real"): colmap[j] = "R5:0"
    elif re.search(r"ln\s*4", h): colmap[j] = "R5:ln4"
    elif re.search(r"λ = 0\.5\b", h): colmap[j] = "F1:0.5"
    elif re.search(r"λ = 0\.75", h): colmap[j] = "F1:0.75"
if len(colmap) != len(t7[0]) - 1:
    stop(f"Table VII header not understood: {t7[0]}")
for r in t7[1:]:
    k = cond_key(r[0].replace("(nominal)", "").strip())
    if k not in fus:
        stop(f"Table VII row not mapped: {r[0]}")
    for j, spec in colmap.items():
        p = fus[k][spec]
        printed = "not reached" if "mean" not in p else f"{p['mean']:.2f} ({p['sd']:.2f})"
        if not r[j].startswith(printed):
            stop(f"Table VII {k} {spec}: computed '{printed}' but the paper prints '{r[j]}'")
        n_checked += 1
data["fusion"] = {"conditions": FUSION_CONDS, "points": fus}

# ---- the paper's other tables, as printed (latency, rank-2 MIMO, varied-IQ training, classical front ends)
tabs = {}
for name in ("II", "IV-a", "IV-b", "VI", "VIII"):
    t = ref["tables"][name]
    tabs[name] = {"header": t[0], "rows": [{"label": r[0], "key": cond_key(r[0]), "cells": [cell(x) for x in r[1:]]} for r in t[1:]]}
data["tables"] = tabs

slot_ms = 1.0 / (cfg["scs_khz"] / 15.0)            # NR numerology: 1 ms subframe, 2^mu slots per subframe
data["meta"] = {k: cfg[k] for k in ("carrier_ghz", "scs_khz", "n_prb", "mcs", "n_rx_ant", "blocks_per_point", "seeds", "llr_sign", "tb_info_bits")}
data["meta"]["slot_ms"] = slot_ms
data["headline"] = {"gain_db": round(gain, 4), "shares_pct": dict(zip(("ln4", "ln8", "ln16"), shares))}
json.dump(data, open(sys.argv[1], "w"), separators=(",", ":"))
print(f"CHECKS PASS: {n_checked} printed operating points (Table V and Table VII), gain {gain:.2f} dB, shares {shares}; "
      f"fusion {sum(len(v) for v in fus.values())} points; tables {', '.join(tabs)}; slot {slot_ms} ms; wrote {sys.argv[1]}")
