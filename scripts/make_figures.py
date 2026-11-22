"""make_figures.py - the three-seed figures (Figs. 3-7, and the seed-42 Fig. 8)
from results/run_c_ci/.

This script reads run_c_ci/seed{42,43,44}/ (and results/minibatch/ for Fig. 3's
L1 arm) and computes every number it draws, except the seed-42 Fig. 8 chart,
which draws the recorded Table VII constants (T7 below). The published Fig. 8
is make_fig08_a3.py's three-seed chart.

Presentation: per arm, the MEAN curve across the three seeds with a shaded
MIN-MAX band. Captions must state that the shaded band spans the three-seed
min-max.

Quoted crossings are the MEAN OF PER-SEED CROSSINGS - the same construction
ci_summary.csv uses - not a crossing read off the mean curve.

GATE: every crossing drawn or annotated must equal the recorded three-seed
value (REF below) within ROUND_DP. Mismatch = STOP that figure.
"""
import csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/run_c_ci"
OUT  = "figures"
SEEDS = (42, 43, 44)
ARMS  = ["0", "ln2", "ln4", "ln8", "ln16", "unbounded"]
ARMLBL = {"0": r"$\Delta_{max}=0$ (= L2-real)", "ln2": r"$\Delta_{max}=\ln 2$",
          "ln4": r"$\Delta_{max}=\ln 4$", "ln8": r"$\Delta_{max}=\ln 8$",
          "ln16": r"$\Delta_{max}=\ln 16$", "unbounded": r"$\Delta_{max}=\infty$ (= R3)"}
DVAL = {"0":0.0,"ln2":math.log(2),"ln4":math.log(4),"ln8":math.log(8),"ln16":math.log(16)}
COND = {  # condition dir -> (irr filename suffix, display label)
    "cdlC_irr25": ("irr25", "CDL-C, IRR = 25 dB (in-dist)"),
    "irr20":      ("irr20", "CDL-C, IRR = 20 dB (5 dB more severe)"),
    "irr35":      ("irr35", "CDL-C, IRR = 35 dB (10 dB milder)"),
    "cdlD":       ("irr25", "CDL-D, IRR = 25 dB (channel shift)"),
    "cdlE":       ("irr25", "CDL-E, IRR = 25 dB (channel shift)"),
}
# Recorded three-seed mean crossings (dB). None = unreachable.
REF = {
 "cdlC_irr25": {"0":20.49,"ln2":19.96,"ln4":19.58,"ln8":19.25,"ln16":19.06,"unbounded":19.05},
 "irr20":      {"0":20.54,"ln2":20.30,"ln4":20.19,"ln8":20.34,"ln16":20.57,"unbounded":None},
 "irr35":      {"0":20.46,"ln2":20.03,"ln4":19.79,"ln8":19.70,"ln16":19.70,"unbounded":22.18},
 "cdlD":       {"0":18.62,"ln2":18.28,"ln4":17.88,"ln8":17.48,"ln16":17.30,"unbounded":17.24},
 "cdlE":       {"0":18.62,"ln2":18.25,"ln4":17.82,"ln8":17.48,"ln16":17.27,"unbounded":17.22},
}
ROUND_DP = 0.02   # tolerance: the recorded values are quoted to 2 dp

def load(cond, arm, seed):
    irr, _ = COND[cond]
    p = f"{ROOT}/seed{seed}/{cond}/r5inf_bler_dmax{arm}_{irr}.csv"
    rows = list(csv.DictReader(open(p)))
    snr = np.array([float(r["snr_db"]) for r in rows])
    bler = np.array([float(r["bler_mean"]) for r in rows])
    nb = int(rows[0]["n_runs"]) * int(rows[0]["n_tti"])
    return snr, bler, nb

def crossing(snr, bler, nb):
    """Corrected detector (reparse_summaries.py convention)."""
    eps = 0.5 / nb
    for i in range(1, len(snr)):
        if bler[i-1] >= 0.1 > bler[i]:
            b1, b2 = max(bler[i-1], eps), max(bler[i], eps)
            l1, l2 = math.log10(b1), math.log10(b2)
            if l1 == l2: return (snr[i-1]+snr[i])/2.0
            return snr[i-1] + (math.log10(0.1)-l1)*(snr[i]-snr[i-1])/(l2-l1)
    return None

def band(cond, arm):
    """Mean curve + min/max band across seeds; and mean-of-per-seed crossings."""
    cur=[]; xs=None; cr=[]
    for s in SEEDS:
        snr,bler,nb = load(cond,arm,s); xs=snr
        cur.append(bler); cr.append(crossing(snr,bler,nb))
    A=np.vstack(cur)
    mean_cross = None if any(c is None for c in cr) else float(np.mean(cr))
    return xs, A.mean(0), A.min(0), A.max(0), mean_cross, cr

def gate(tag):
    """Check every computed crossing against the recorded values (REF)."""
    bad=[]
    for cond in COND:
        for arm in ARMS:
            _,_,_,_,mc,_ = band(cond,arm)
            ref = REF[cond][arm]
            if ref is None:
                if mc is not None: bad.append(f"{cond}/{arm}: computed {mc:.4f}, recorded unreachable")
            elif mc is None:
                bad.append(f"{cond}/{arm}: computed unreachable, recorded {ref}")
            elif abs(mc-ref) > ROUND_DP:
                bad.append(f"{cond}/{arm}: computed {mc:.4f} vs recorded {ref} (|d|={abs(mc-ref):.4f} > {ROUND_DP})")
    print(f"  [{tag}] gate: {'PASS - all crossings match the recorded values' if not bad else 'FAIL'}")
    for b in bad: print("       ", b)
    return not bad

BANDNOTE = "shaded band spans the three-seed min-max"

def fig_frontier():           # Fig. 6  fig01_headline_gain_vs_safety
    xs=[]; ys=[]; lo=[]; hi=[]
    for arm in ARMS[:-1]:
        _,_,_,_,mc,cr = band("cdlC_irr25",arm)
        xs.append(DVAL[arm]); ys.append(mc); lo.append(min(cr)); hi.append(max(cr))
    _,_,_,_,unb,cru = band("cdlC_irr25","unbounded")
    l2, ln4 = ys[0], ys[2]
    share = (l2-ln4)/(l2-unb)*100.0                     # COMPUTED, not hardcoded
    f,ax=plt.subplots(figsize=(3.5,2.6),dpi=300)
    ax.fill_between(xs,lo,hi,alpha=.22,color="tab:blue",lw=0)
    ax.plot(xs,ys,"o-",color="tab:blue",ms=4,label="bounded R5")
    ax.axhline(l2,ls=":",c="0.45",lw=.9); ax.axhline(unb,ls="--",c="tab:red",lw=.9)
    ax.text(xs[-1],l2,"  L2-real baseline",va="bottom",ha="right",fontsize=6,color="0.35")
    ax.text(xs[-1],unb,"  R3 (unbounded)",va="top",ha="right",fontsize=6,color="tab:red")
    ax.annotate(f"$\\ln 4$: {share:.0f}% of gain",xy=(DVAL['ln4'],ln4),
                xytext=(DVAL['ln4']+.25,ln4+.28),fontsize=6,
                arrowprops=dict(arrowstyle="->",lw=.6))
    ax.set_xlabel(r"Bound $\Delta_{max}$ (nats)"); ax.set_ylabel(r"10% BLER $E_b/N_0$ (dB)")
    ax.invert_yaxis(); ax.grid(alpha=.3,lw=.4); f.tight_layout()
    f.savefig(f"{OUT}/fig01_headline_gain_vs_safety.png"); f.savefig(f"{OUT}/fig01_headline_gain_vs_safety.pdf"); plt.close(f)
    return {"ln4_share_pct":share,"L2":l2,"ln4":ln4,"R3":unb}

def fig_bler_sweep():         # Fig. 7  fig02_bler_vs_snr_dmax_sweep
    f,ax=plt.subplots(figsize=(3.5,2.7),dpi=300)
    for arm in ARMS:
        xs,mu,lo,hi,_,_ = band("cdlC_irr25",arm)
        ax.fill_between(xs,np.maximum(lo,1e-4),np.maximum(hi,1e-4),alpha=.16,lw=0)
        ax.semilogy(xs,np.maximum(mu,1e-4),marker="o",ms=2.6,lw=1,label=ARMLBL[arm])
    ax.axhline(.1,ls=":",c="0.4",lw=.8); ax.text(xs[0],.11,"10% BLER",fontsize=6,color="0.4")
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("BLER")
    ax.grid(alpha=.3,lw=.4,which="both"); ax.legend(fontsize=5,loc="lower left"); f.tight_layout()
    f.savefig(f"{OUT}/fig02_bler_vs_snr_dmax_sweep.png"); f.savefig(f"{OUT}/fig02_bler_vs_snr_dmax_sweep.pdf"); plt.close(f)

def fig_two_axis():           # Fig. 4  fig03_two_axis_ood
    f,ax=plt.subplots(figsize=(3.6,2.8),dpi=300)
    for cond,(_,lbl) in COND.items():
        xs=[];ys=[];lo=[];hi=[];unre=[]
        for arm in ARMS[:-1]:
            _,_,_,_,mc,cr=band(cond,arm); xs.append(DVAL[arm]); ys.append(mc)
            lo.append(min(cr)); hi.append(max(cr))
        ax.fill_between(xs,lo,hi,alpha=.14,lw=0)
        ax.plot(xs,ys,marker="o",ms=3,lw=1,label=lbl)
        _,_,_,_,mu,cru=band(cond,"unbounded")
        if mu is None: ax.plot([DVAL["ln16"]+.35],[max(ys)],"x",ms=7,mew=1.6,color="tab:red")
    ax.set_xlabel(r"Bound $\Delta_{max}$ (nats)"); ax.set_ylabel(r"10% BLER $E_b/N_0$ (dB)")
    ax.invert_yaxis(); ax.grid(alpha=.3,lw=.4)
    ax.legend(fontsize=5,loc="lower center",bbox_to_anchor=(.5,1.03),ncol=2,
              borderaxespad=0,frameon=True,handlelength=1.6,columnspacing=1.2)
    f.tight_layout(); f.savefig(f"{OUT}/fig03_two_axis_ood.png",bbox_inches="tight")
    f.savefig(f"{OUT}/fig03_two_axis_ood.pdf",bbox_inches="tight"); plt.close(f)

def fig_irr_failure():        # Fig. 5  fig04_r3_irr_shift_failure
    f,axes=plt.subplots(1,2,figsize=(6.6,2.7),dpi=300,sharey=True)
    for ax,(cond,title) in zip(axes,[("irr20","IRR = 20 dB (5 dB more severe)"),
                                     ("irr35","IRR = 35 dB (10 dB milder)")]):
        for arm in ["0","ln4","ln8","unbounded"]:
            xs,mu,lo,hi,_,_=band(cond,arm)
            ax.fill_between(xs,np.maximum(lo,1e-4),np.maximum(hi,1e-4),alpha=.16,lw=0)
            ax.semilogy(xs,np.maximum(mu,1e-4),marker="o",ms=2.6,lw=1,label=ARMLBL[arm])
        ax.axhline(.1,ls=":",c="0.4",lw=.8); ax.set_title(title,fontsize=7)
        ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.grid(alpha=.3,lw=.4,which="both")
    axes[0].set_ylabel("BLER"); axes[1].legend(fontsize=5,loc="lower left")
    f.tight_layout(); f.savefig(f"{OUT}/fig04_r3_irr_shift_failure.png")
    f.savefig(f"{OUT}/fig04_r3_irr_shift_failure.pdf"); plt.close(f)

if __name__ == "__main__":
    os.makedirs(OUT,exist_ok=True)
    if not gate("pre-draw"): sys.exit("GATE FAILED - no figure written")
    d=fig_frontier(); print(f"  Fig 6 written; computed ln4 share = {d['ln4_share_pct']:.1f}%")
    fig_bler_sweep(); print("  Fig 7 written")
    fig_two_axis();   print("  Fig 4 written")
    fig_irr_failure();print("  Fig 5 written")
    print(f"  BAND NOTE for captions: '{BANDNOTE}'")

MB = "results/minibatch"

def _simo_l1_min():
    """L1 arm from the mini-batch: unreachable; return the min over the evaluated range."""
    mins=[]
    for s in SEEDS:
        p=f"{MB}/simo_l1/seed{s}/bler_cdlC_mcs27_nadd1_v3.csv"
        rows=list(csv.DictReader(open(p)))
        bl=[float(r["bler_mean"]) for r in rows]; sn=[float(r["snr_db"]) for r in rows]
        nb=int(rows[0]["n_runs"])*int(rows[0]["n_blocks_per_run"])
        assert crossing(np.array(sn),np.array(bl),nb) is None, "L1 unexpectedly crosses 10%"
        mins.append(min(bl))
    return float(np.mean(mins)), sn[int(np.argmin(bl))]

def fig_attribution():        # Fig. 3  fig05_classical_baseline_attribution
    l1_min, l1_at = _simo_l1_min()
    _,_,_,_,l2,_  = band("cdlC_irr25","0")
    _,_,_,_,ln4,_ = band("cdlC_irr25","ln4")
    _,_,_,_,r3,_  = band("cdlC_irr25","unbounded")
    # gate against the recorded three-seed means before drawing
    for got,want,tag in ((l2,20.49,"L2"),(ln4,19.58,"ln4"),(r3,19.05,"R3")):
        assert abs(got-want)<=ROUND_DP, f"Fig.3 gate {tag}: {got} vs {want}"
    assert abs((l2-r3)-1.44)<0.02 and abs((l2-ln4)-0.91)<0.02, "Fig.3 bracket gate"
    # gain over the fair baseline; L2-real is the 0 dB reference
    g_ln4, g_r3 = l2-ln4, l2-r3
    share = 100.0*g_ln4/g_r3
    assert abs(share-63.0) < 1.5, f"Fig.3 share gate: {share:.1f}%"
    # Colours follow Fig. 7's Delta_max cycle: L2-real C0, ln 4 C2, R3 C5; L1 grey.
    xs     = [0,1,2,3]
    heights= [0.0, g_ln4, g_r3, -0.55]          # L1 drawn below the axis (not reached)
    cols   = ["tab:blue","tab:green","tab:brown","0.6"]
    labels = ["R1-L2-real\n(fair baseline)", r"R5, $\Delta_{\mathrm{max}}=\ln 4$",
              "R3\n(unconstrained)", "R1-L1\n(no IQI comp.)"]
    cross  = [l2, ln4, r3, None]
    f,ax=plt.subplots(figsize=(3.5,2.45),dpi=300)
    for xi,h,c in zip(xs,heights,cols):
        ax.bar(xi,h,0.56,color=c,edgecolor="0.25",linewidth=.5,
               hatch=("///" if h<0 else None),zorder=2)
    # bold zero line = the baseline
    ax.axhline(0,color="0.15",lw=1.5,zorder=3)
    ax.text(3.46,0.035,"L2-real baseline",ha="right",va="bottom",fontsize=5.4,color="0.15")
    # absolute crossings above each reachable bar
    for xi,h,c,cr in zip(xs,heights,cols,cross):
        if cr is None: continue
        ax.text(xi,h+0.055,f"{cr:.2f} dB",ha="center",va="bottom",fontsize=5.8,color=c)
    # the bounded bar's share of the unconstrained gain
    ax.text(1,g_ln4/2,"63% of\nR3 gain",ha="center",va="center",fontsize=5.4,color="white",zorder=4)
    ax.annotate("not reached\n(min. BLER %.2f)"%l1_min,(3,-0.55),xytext=(0,-4),
                textcoords="offset points",ha="center",va="top",fontsize=5.4,color="0.3")
    ax.set_xticks(xs); ax.set_xticklabels(labels,fontsize=5.8)
    ax.set_ylabel("Gain over L2-real at 10% BLER (dB)",fontsize=6.4)
    ax.set_ylim(-1.05,1.85); ax.set_xlim(-0.6,3.6)
    ax.tick_params(axis="y",labelsize=6); ax.tick_params(axis="x",length=0)
    ax.grid(axis="y",alpha=.3,lw=.4); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    f.tight_layout()
    f.savefig(f"{OUT}/fig05_classical_baseline_attribution.png",bbox_inches="tight")
    f.savefig(f"{OUT}/fig05_classical_baseline_attribution.pdf",bbox_inches="tight"); plt.close(f)
    return {"L1_min":l1_min,"L1_at":l1_at,"L2":l2,"ln4":ln4,"R3":r3}


# ---------------------------------------------------------------- Fig. 8
# Fusion rules against the bound, as gain over the classical baseline.
# Run A (seed 42, paired on shared realizations). Every input is gated
# against the recorded Table VII values before drawing.
T7 = {  # condition -> (L2-real, R5 ln4, F1 lambda=0.5, F1 lambda=0.75)
 "CDL-C\nIRR 25 dB\n(nominal)":       (20.4490, 19.5950, 19.1034, 18.7376),
 "CDL-C\nIRR 20 dB\n(impair. shift)": (20.4912, 20.2240, 19.9766, 20.8189),
 "CDL-C\nIRR 35 dB\n(impair. shift)": (20.4344, 19.8370, 19.3337, 19.4574),
 "CDL-D\nIRR 25 dB\n(channel shift)": (18.5909, 17.8540, 17.1515, 16.6995),
 "CDL-E\nIRR 25 dB\n(channel shift)": (18.5831, 17.8140, 17.1778, 16.6710),
}

def fig_fusion_vs_bound():    # Fig. 8  fig08_fusion_vs_bound
    conds = list(T7)
    n, s20 = T7[conds[0]], T7[conds[1]]        # gate the two margins in the text
    assert abs((n[1] - n[3]) - 0.857) < 1e-3, "nominal margin gate"
    assert abs((s20[0] - s20[3]) + 0.328) < 1e-3, "IRR-20 regression gate"
    # ln 4 keeps Fig. 7's Delta_max colour (C2); the F1 arms take colours
    # outside that cycle so no reader maps them onto a Delta_max value.
    arms = [(r"R5, $\Delta_{\mathrm{max}}=\ln 4$",       1, "tab:green"),
            (r"convex interpolation, $\lambda=0.5$",  2, "tab:olive"),
            (r"convex interpolation, $\lambda=0.75$", 3, "tab:cyan")]
    f, ax = plt.subplots(figsize=(3.5, 2.60), dpi=300)
    x = np.arange(len(conds)); w = 0.26
    for k, (lbl, idx, col) in enumerate(arms):
        g = [T7[c][0] - T7[c][idx] for c in conds]
        ax.bar(x + (k - 1) * w, g, w, label=lbl, color=col,
               edgecolor="0.25", linewidth=.4)
        for xi, gi in zip(x + (k - 1) * w, g):
            if gi < 0:
                ax.annotate("−0.33 dB", (xi, gi), textcoords="offset points",
                            xytext=(0, -8), ha="center", fontsize=5.6, color="0.15")
    ax.axhline(0, color="0.1", lw=1.2, zorder=3)
    # right end of the zero line, just below it: the only region clear of every
    # bar (the sole negative bar sits at the second condition, far to the left)
    ax.text(len(conds) - 0.46, -0.06, "classical baseline (L2-real)", fontsize=5.4,
            color="0.1", va="top", ha="right", zorder=4)
    ax.set_xlim(-0.6, len(conds) - 0.4); ax.set_ylim(-0.62, 2.15)
    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=5.0)
    ax.set_ylabel("Gain over L2-real at 10% BLER (dB)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=.3, lw=.4)
    ax.legend(fontsize=5.0, loc="lower center", bbox_to_anchor=(.5, 1.005),
              ncol=3, borderaxespad=0, frameon=True, handlelength=1.2,
              columnspacing=0.7, handletextpad=0.4)
    f.tight_layout()
    f.savefig(f"{OUT}/fig08_fusion_vs_bound.png", bbox_inches="tight")
    f.savefig(f"{OUT}/fig08_fusion_vs_bound.pdf", bbox_inches="tight"); plt.close(f)

if __name__ == "__main__":
    a=fig_attribution(); print(f"  Fig 3 written; L1 min BLER {a['L1_min']:.2f}, crossings {a['L2']:.2f}/{a['ln4']:.2f}/{a['R3']:.2f}")
    fig_fusion_vs_bound(); print("  Fig 8 written")
