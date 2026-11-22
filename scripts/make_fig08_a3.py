"""make_fig08_a3.py - Fig. 8 from run A3's three seeds.

Same chart as make_figures.py's fig_fusion_vs_bound (the seed-42 record, left untouched): the gain over the
classical baseline (L2-real) at 10% BLER of R5 at Delta_max = ln 4 and of convex interpolation at lambda = 0.5 and 0.75,
on the five conditions of Table VII. Now each bar is the three-seed mean of the per-seed paired gain and its whisker
spans the three-seed min-max (the band convention of Figs. 4-7). Every crossing is recomputed at full precision from
run A3's per-point shards with run A3's own detector (run_a3.crossing, verbatim); nothing is hard-coded.

Gates (STOP on mismatch): every per-seed crossing equals results/run_a3/seed<S>/a3_summary.csv at 4 dp; the two margins
the prose prints reproduce at 2 dp (nominal R5 ln 4 - F1 0.75 = 0.93; IRR 20 L2 - F1 0.75 = -0.35).
Colours: ln 4 keeps C2 (tab:green), the two lambda arms tab:olive / tab:cyan. Size: 1036 x 749 px at 300 dpi,
the pixel size of the media part it replaces (word/media/image8.png, extent cx 2977709 cy 2152803), so the drawing's
extent and the pagination hold. draw(..., baseline_label=False) drops the floating zero-line label; the
default output is unchanged.

  TF_USE_LEGACY_KERAS=1 python make_fig08_a3.py      (paper3 env; from the folder that holds results/)
"""
import csv, json, math, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A3 = "results/run_a3/seed{}"
OUT = "figures/fig08_fusion_vs_bound_a3"
SEEDS = (42, 43, 44)
CONDS = [("cdlC_irr25", "CDL-C\nIRR 25 dB\n(nominal)"), ("irr20", "CDL-C\nIRR 20 dB\n(impair. shift)"),
         ("irr35", "CDL-C\nIRR 35 dB\n(impair. shift)"), ("cdlD", "CDL-D\nIRR 25 dB\n(channel shift)"),
         ("cdlE", "CDL-E\nIRR 25 dB\n(channel shift)")]


def crossing(snr, errors, n):                  # run_a3.crossing, verbatim
    b = [e / n for e in errors]
    for i in range(1, len(snr)):
        if b[i - 1] >= 0.1 > b[i] and b[i - 1] > 0 and b[i] > 0:
            l1, l2 = math.log10(b[i - 1]), math.log10(b[i])
            return snr[i - 1] + (math.log10(0.1) - l1) * (snr[i] - snr[i - 1]) / (l2 - l1)
    return None


def cross_of(path):
    sh = json.load(open(path))
    return crossing(sh["snr_db"], sh["errors"], sh["n_blocks"])


def load():
    v = {}
    for s in SEEDS:
        summ = {(r["condition"], r["kind"], r["param"]): r["crossing_db"] for r in csv.DictReader(open(f"{A3.format(s)}/a3_summary.csv"))}
        for c, _ in CONDS:
            got = {"L2": cross_of(f"{A3.format(s)}/ag0_shards/{c}__0.json"),
                   "ln4": cross_of(f"{A3.format(s)}/ag0_shards/{c}__ln4.json"),
                   "f05": cross_of(f"{A3.format(s)}/shards/{c}__F1__0.5.json"),
                   "f075": cross_of(f"{A3.format(s)}/shards/{c}__F1__0.75.json")}
            for k, key in (("L2", ("R5", "0")), ("ln4", ("R5", "ln4")), ("f05", ("F1", "0.5")), ("f075", ("F1", "0.75"))):
                assert got[k] is not None, f"{s} {c} {k} unreachable"
                assert f"{got[k]:.4f}" == summ[(c,) + key], f"gate: seed {s} {c} {k} {got[k]:.4f} vs a3_summary {summ[(c,) + key]}"
            v[(s, c)] = got
    return v


def gates(v):
    nom = st.mean(v[(s, "cdlC_irr25")]["ln4"] - v[(s, "cdlC_irr25")]["f075"] for s in SEEDS)
    reg = st.mean(v[(s, "irr20")]["L2"] - v[(s, "irr20")]["f075"] for s in SEEDS)
    assert f"{nom:.2f}" == "0.93", f"nominal margin gate {nom}"
    assert f"{reg:.2f}" == "-0.35", f"IRR-20 regression gate {reg}"
    return nom, reg


def draw(v, out=OUT, baseline_label=True, pdf=True):
    """The chart; baseline_label=False drops the floating "classical baseline (L2-real)" text and
    keeps the thick zero line and the negative-bar annotations. Returns the bar rows."""
    arms = [(r"R5, $\Delta_{\mathrm{max}}=\ln 4$", "ln4", "tab:green"),
            (r"convex interpolation, $\lambda=0.5$", "f05", "tab:olive"),
            (r"convex interpolation, $\lambda=0.75$", "f075", "tab:cyan")]
    f, ax = plt.subplots(figsize=(1036 / 300, 749 / 300), dpi=300)
    x = np.arange(len(CONDS)); w = 0.26
    rows = []
    for k, (lbl, key, col) in enumerate(arms):
        g = [[v[(s, c)]["L2"] - v[(s, c)][key] for s in SEEDS] for c, _ in CONDS]
        m = [st.mean(gi) for gi in g]; lo = [mi - min(gi) for mi, gi in zip(m, g)]; hi = [max(gi) - mi for mi, gi in zip(m, g)]
        xs = x + (k - 1) * w
        ax.bar(xs, m, w, label=lbl, color=col, edgecolor="0.25", linewidth=.4)
        ax.errorbar(xs, m, yerr=[lo, hi], fmt="none", ecolor="0.15", elinewidth=.5, capsize=1.2, capthick=.5)
        for xi, mi, gi in zip(xs, m, g):
            if mi < 0:
                ax.annotate(f"−{abs(mi):.2f} dB", (xi, min(gi)), textcoords="offset points", xytext=(0, -8),
                            ha="center", fontsize=5.6, color="0.15")
        rows += [(c, key, mi, min(gi), max(gi)) for (c, _), mi, gi in zip(CONDS, m, g)]
    ax.axhline(0, color="0.1", lw=1.2, zorder=3)
    if baseline_label:
        ax.text(len(CONDS) - 0.46, -0.06, "classical baseline (L2-real)", fontsize=5.4, color="0.1", va="top", ha="right", zorder=4)
    ax.set_xlim(-0.6, len(CONDS) - 0.4); ax.set_ylim(-0.62, 2.15)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in CONDS], fontsize=5.0)
    ax.set_ylabel("Gain over L2-real at 10% BLER (dB)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=.3, lw=.4)
    ax.legend(fontsize=5.0, loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=3, borderaxespad=0, frameon=True,
              handlelength=1.2, columnspacing=0.7, handletextpad=0.4)
    f.tight_layout(pad=0.3)
    f.savefig(f"{out}.png", dpi=300)
    if pdf: f.savefig(f"{out}.pdf")
    plt.close(f)
    with open(f"{out}_bars.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["condition", "arm", "gain_mean_db", "gain_min_db", "gain_max_db"])
        for r in rows: wr.writerow([r[0], r[1]] + [f"{t:.4f}" for t in r[2:]])
    return rows


def main():
    v = load(); nom, reg = gates(v)
    draw(v)
    print(f"Fig. 8 (A3) written: {OUT}.png; gates PASS (nominal margin {nom:.4f} -> 0.93, IRR-20 regression {reg:.4f} -> -0.35)")


if __name__ == "__main__":
    main()
