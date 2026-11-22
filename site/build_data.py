"""build_data.py OUT.json - the data the page shows, built from results/ in this repository.

Every 10% error point and its cross-seed spread is checked against tests/reference_values.json (Table V of the published
paper, as printed), and so are the headline gain and the shares of it kept at ln 4, ln 8 and ln 16. Any mismatch stops the
build and nothing is written.

    python site/build_data.py build/data.json
"""
import csv, json, math, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DM = ["0", "ln2", "ln4", "ln8", "ln16", "unbounded"]
SEEDS = (42, 43, 44)
SHORT = {"cdlC_irr25": "Training conditions", "cdlD": "New channel: CDL-D", "cdlE": "New channel: CDL-E",
         "irr20": "Harsher IQ imbalance", "irr35": "Milder IQ imbalance", "joint": "Both shifts at once"}


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


cfg = json.load(open(os.path.join(RES, "run_config.json")))
slots = json.load(open(os.path.join(RES, "llr_slots.json")))
ref = json.load(open(os.path.join(ROOT, "tests", "reference_values.json")))
if ref["dmax"] != DM:
    stop("the reference values use a different bound order")

data = {"conditions": [], "dmax": DM}
for key, short in SHORT.items():
    cc = cfg["conditions"][key]
    irr = int(cc["irr_db"]) if float(cc["irr_db"]).is_integer() else cc["irr_db"]
    entry = {"key": key, "short": short, "detail": f"CDL-{cc['channel']} channel, IRR {irr} dB", "curves": {}, "crossing": {}}
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
                              "hi": [round(max(c), 6) for c in cols]}
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
        cell = ref["table_v"][key][DM.index(d)]
        if not cell.startswith(printed):
            stop(f"{key} {d}: computed '{printed}' but the paper prints '{cell}'")
    entry["slot"] = slots[key]
    data["conditions"].append(entry)

c = data["conditions"][0]["crossing"]
gain = c["0"]["mean"] - c["unbounded"]["mean"]
shares = [round(100 * (c["0"]["mean"] - c[d]["mean"]) / gain) for d in ("ln4", "ln8", "ln16")]
if f"{gain:.2f}" != ref["headline_gain_db"] or shares != ref["shares_pct"]:
    stop(f"headline {gain:.2f} dB / {shares} differs from the paper's {ref['headline_gain_db']} dB / {ref['shares_pct']}")
data["meta"] = {k: cfg[k] for k in ("carrier_ghz", "scs_khz", "n_prb", "mcs", "n_rx_ant", "blocks_per_point", "seeds", "llr_sign")}
data["headline"] = {"gain_db": round(gain, 4), "shares_pct": dict(zip(("ln4", "ln8", "ln16"), shares))}
json.dump(data, open(sys.argv[1], "w"), separators=(",", ":"))
print(f"CHECKS PASS: 36 Table V cells (mean and SD), gain {gain:.2f} dB, shares {shares}; wrote {sys.argv[1]}")
