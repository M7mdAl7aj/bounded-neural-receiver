"""run_b2_clipstats.py - Run B2 clip statistics. Offline on the LLR dumps
(r5_simo_inference.py --dump-llrs), CPU only, under the locked paper3
environment (numpy 1.26.4; gate B2-G5).

Built on run_b_clipdiag.py's structure: chunked (CHUNK = 128 blocks),
memory-mapped, exact 20,000-bin log-spaced histograms (1e-6..1e4), the LLR
sign convention (positive LLR encodes bit 1; the hard decision is L > 0). Every
quantity is an exact count or an exact-histogram quantile; nothing is sampled
except the synthetic streams of gate B2-G3.

  a = L_c (L2-real, llr_l2), b = L_R3 (llr_r3), tb = true coded bit
  clip event at Delta_max:  |b - a| > Delta_max ;  L_R5 = clip(b, a +/- Delta_max)

Quantities, per (seed, condition, SNR point) and pooled over 16-24 dB:
  1  residual quantiles of |b - a| (p10 p25 p50 p75 p90 p99 p99.9)
  2  clip fraction (and upper/lower split) at each Delta_max
  3  sign disagreement (a > 0) != (b > 0): all bits; among clipped bits
  4  among clipped bits, the 2x2 of R3 sign right/wrong x L_c sign right/wrong
     against tb. Marginals: wrong-direction share = R3 wrong (run B's
     wrong_direction_share); anchor-error share = L_c wrong.
     Restricted 2x2: sign-disagreeing bits with |L_c| > Delta_max, where
     Theorem 1's sign preservation gives L_R5 the classical sign: (R3 right,
     L_c wrong) = correct neural decisions the clip reverses; (R3 wrong, L_c
     right) = wrong neural decisions it repairs; their ratio.
  5  among clipped bits: sign L_R5 != sign L_R3 (the clip reverses the neural
     decision), sign L_R5 != sign L_c, same-sign trims vs sign-differing
  6  |b| / |a|: quantiles over all bits and among clipped bits, overall and in
     run B's four |L_c| quartile bins (clip rate per bin reproduces run B)
  7  hard-decision error rate of L_c, L_R3, L_R5: all bits; among clipped bits
  8a ECE: 15 equal-mass bins of stated confidence sigma(|L|) (formed on |L|,
     which orders bits identically and avoids float ties of sigma at 1.0),
     ECE, MCE, reliability table. Secondary: bounded by the error rate.
  8b LLR-scale reliability: per |L| bin (run B's 20,000 edges aggregated to
     40 bins per decade) n, n_correct, mean stated |L| and the Jeffreys-
     corrected empirical log-odds ln((n_c + 0.5)/(n - n_c + 0.5)); calibration
     slope = weighted least squares through the origin of log-odds on stated
     |L| over bins with >= 100 errors, weights = inverse delta-method variance
     1/(1/(n_c + 0.5) + 1/(n - n_c + 0.5)); log-odds interpolated at
     |L| = 1, 3, 10, 30, 100 within the measurable range; the largest |L| bin
     still holding 100 errors.
  9  every quantity per SNR point as well as pooled
 10  all five conditions, including CDL-D and CDL-E

Gates: B2-G5, G1, G2, G3 (relative form), G4. Outputs other than gates.csv
are written only in --mode full and only if every gate passed.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, csv, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Exact reproduction of run B needs the locked stack: numpy 2.x computes
# np.logspace differently in the last bit for 4,903 of the 20,002 histogram
# edges, which moves the residual quantiles by one ulp. Run with the locked
# paper3 environment's python (numpy 1.26.4).
if np.__version__ != "1.26.4":
    sys.exit(f"run_b2_clipstats.py: numpy {np.__version__}; run under the paper3 env (numpy 1.26.4)")

PKG = os.path.dirname(os.path.abspath(__file__))
DUMPS = f"{PKG}/results/run_c_ci/dumps"
JOINT = {(42, "cdlD_irr20"): f"{PKG}/results/run_j/dumps/seed42/cdlD_irr20"}   # the seed-42 joint set


def dump_dir(seed, cond):
    return JOINT.get((seed, cond), f"{DUMPS}/seed{seed}/{cond}")
OUT = f"{PKG}/results/run_b2_clipstats"
RUNB = f"{PKG}/results/run_b_clipdiag/clipdiag_summary.csv"
RUNB_HIST_L2 = f"{PKG}/results/run_b_clipdiag/hist_absl2_cdlC_irr25.npy"
CONDS = ["cdlC_irr25", "irr20", "irr35", "cdlD", "cdlE"]
RUNB_CONDS = ["cdlC_irr25", "irr20", "irr35"]
DMAX = [("0", 0.0), ("ln2", math.log(2)), ("ln4", math.log(4)),
        ("ln8", math.log(8)), ("ln16", math.log(16)), ("unbounded", math.inf)]
FINITE = [(n, v) for n, v in DMAX if np.isfinite(v)]
CHUNK = 128
NB = 20000
EDGES = np.concatenate([[0.0], np.logspace(-6, 4, NB), [np.inf]])
NFINE = len(EDGES) - 1                       # 20,001 fine bins, [e_i, e_i+1)
_lo = EDGES[:-1]
CMAP = (np.where(_lo > 0, np.floor((np.log10(np.where(_lo > 0, _lo, 1.0)) + 6.0) * 40.0), -1.0)
        .astype(np.int64) + 1)               # coarse bin 0 = |L| < 1e-6; 40 per decade above
NCOARSE = int(CMAP.max()) + 1
C_LO = np.array([EDGES[np.argmax(CMAP == c)] for c in range(NCOARSE)])
C_HI = np.append(C_LO[1:], np.inf)
NR = 48000                                   # log10 ratio grid over [-12, 12]
RLO, RHI = -12.0, 12.0
NCAL = 15
RQ = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999]
RQN = ["p10", "p25", "p50", "p75", "p90", "p99", "p999"]
RATQ = [0.10, 0.25, 0.50, 0.75, 0.90]
LTARGETS = [1, 3, 10, 30, 100]
MIN_ERR = 100
SYN_N, SYN_SEED = 10_000_000, 20261016


def hist_quantiles(counts, edges, qs):          # verbatim from run_b_clipdiag.py
    tot = counts.sum()
    if tot == 0: return [float("nan")] * len(qs)
    c = np.cumsum(counts)
    return [float(edges[min(int(np.searchsorted(c, q * tot)), len(edges) - 2)])
            for q in qs]


def ratio_hist(x):
    """Histogram of log10 |b|/|a| on [RLO, RHI] plus [under incl. -inf, over incl. +inf, nan]."""
    h = np.histogram(x[np.isfinite(x)], bins=NR, range=(RLO, RHI))[0].astype(np.int64)
    extra = np.array([np.count_nonzero(x < RLO), np.count_nonzero(x > RHI),
                      np.count_nonzero(np.isnan(x))], dtype=np.int64)
    return np.concatenate([h, extra])


def ratio_quantiles(h, qs):
    core, under, over = h[:NR], h[NR], h[NR + 1]
    cnt = np.concatenate([[under], core, [over]])
    tot = cnt.sum()
    if tot == 0: return [float("nan")] * len(qs)
    edges = np.concatenate([[-np.inf], np.linspace(RLO, RHI, NR + 1)])
    c = np.cumsum(cnt)
    out = []
    for q in qs:
        i = min(int(np.searchsorted(c, q * tot)), len(edges) - 1)
        out.append(float(10.0 ** edges[i]) if np.isfinite(edges[i]) else 0.0)
    return out


def fmt(v):
    """Same text a csv.DictWriter writes for a float or int (run B's CSV)."""
    return str(int(v)) if isinstance(v, (int, np.integer)) else str(float(v))


def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def tags_of(d):
    m = json.load(open(os.path.join(d, "index.json")))
    return m, [f"snr{s:.1f}" for s in m["snr_db"]], [float(s) for s in m["snr_db"]]


def chunks(d, t):
    l2 = np.load(f"{d}/llr_l2_{t}.npy", mmap_mode="r")
    r3 = np.load(f"{d}/llr_r3_{t}.npy", mmap_mode="r")
    bc = np.load(f"{d}/bits_coded_{t}.npy", mmap_mode="r")
    for lo in range(0, l2.shape[0], CHUNK):
        yield (np.asarray(l2[lo:lo + CHUNK], dtype=np.float64).ravel(),
               np.asarray(r3[lo:lo + CHUNK], dtype=np.float64).ravel(),
               np.asarray(bc[lo:lo + CHUNK]).ravel())


def streams_of(a, b):
    """The calibrated LLR streams: L_c, L_R3 and L_R5 at each finite Delta_max > 0."""
    s = {"Lc": a, "LR3": b}
    for n, v in FINITE:
        if v > 0:
            s[f"LR5_{n}"] = np.clip(b, a - v, a + v)
    return s


STREAMS = ["Lc", "LR3"] + [f"LR5_{n}" for n, v in FINITE if v > 0]


# ---------------- pass 1: magnitude histograms (quantile edges) ----------------
def pass1(job):
    seed, cond, t = job
    d = dump_dir(seed, cond)
    h = {"d": np.zeros(NFINE, np.int64)}
    for a, b, _ in chunks(d, t):
        h["d"] += np.histogram(np.abs(b - a), bins=EDGES)[0]
        for k, x in streams_of(a, b).items():
            h.setdefault(k, np.zeros(NFINE, np.int64))
            h[k] += np.histogram(np.abs(x), bins=EDGES)[0]
    return job, h


def cal_edges(hist):
    e = hist_quantiles(hist, EDGES, [k / NCAL for k in range(1, NCAL)])
    return np.unique(np.array(e, dtype=np.float64))


# ---------------- pass 2: every count ----------------
def pass2(job):
    (seed, cond, t), l2q, e_unit, e_pool = job
    d = dump_dir(seed, cond)
    nd = len(DMAX)
    z = lambda *s: np.zeros(s, np.int64)
    r = dict(n=0, sd_all=0, err_lc=0, err_r3=0,
             n_clip=z(nd), n_up=z(nd), n_lo=z(nd), mismatch=z(nd), val_dis0=0,
             sd_clip=z(nd), j=z(nd, 4), rev53=z(nd), dif5c=z(nd), same_clip=z(nd),
             err_r5=z(nd), err_r5_clip=z(nd), bins_tot=z(4), bins_clip=z(nd, 4),
             res=z(nd, 4), res_indep=z(nd), res_notclip=z(nd), res_r5_not_lc=z(nd),
             rat_all=z(NR + 3), rat_all_q=z(4, NR + 3),
             rat_clip=z(nd, NR + 3), rat_clip_q=z(nd, 4, NR + 3))
    for k in STREAMS:
        r[f"rel_n_{k}"] = np.zeros(NFINE, np.int64)
        r[f"rel_c_{k}"] = np.zeros(NFINE, np.int64)
        r[f"rel_L_{k}"] = np.zeros(NFINE, np.float64)
    cal = {}
    for a, b, tb in chunks(d, t):
        n = a.size
        r["n"] += n
        sa, sb = (a > 0), (b > 0)
        bit = tb.astype(bool)
        ra, rb = (sa == bit), (sb == bit)
        r["err_lc"] += int(n - ra.sum()); r["err_r3"] += int(n - rb.sum())
        sdis = sa != sb
        r["sd_all"] += int(sdis.sum())
        diff = b - a
        ad = np.abs(diff)
        absa = np.abs(a)
        idx = np.digitize(absa, l2q)                     # run B's |L_c| quartile bins
        r["bins_tot"] += np.bincount(idx, minlength=4)[:4]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log10(np.abs(b)) - np.log10(absa)
        r["rat_all"] += ratio_hist(lr)
        for k in range(4):
            r["rat_all_q"][k] += ratio_hist(lr[idx == k])
        r["val_dis0"] += int(np.count_nonzero(b != a))
        for i, (name, v) in enumerate(DMAX):
            clip = ad > v if np.isfinite(v) else np.zeros(n, bool)
            r5 = np.clip(b, a - v, a + v) if np.isfinite(v) else b
            s5 = r5 > 0
            r["mismatch"][i] += int(np.count_nonzero(clip != (r5 != b)))
            r["n_clip"][i] += int(clip.sum())
            r["n_up"][i] += int((clip & (diff > 0)).sum())
            r["n_lo"][i] += int((clip & (diff < 0)).sum())
            r["sd_clip"][i] += int((clip & sdis).sum())
            r["j"][i] += [int((clip & rb & ra).sum()), int((clip & rb & ~ra).sum()),
                          int((clip & ~rb & ra).sum()), int((clip & ~rb & ~ra).sum())]
            r["rev53"][i] += int((clip & (s5 != sb)).sum())
            r["dif5c"][i] += int((clip & (s5 != sa)).sum())
            r["same_clip"][i] += int((clip & ~sdis).sum())
            r5right = (s5 == bit)
            r["err_r5"][i] += int(n - r5right.sum())
            r["err_r5_clip"][i] += int((clip & ~r5right).sum())
            rmask = sdis & (absa > v) if np.isfinite(v) else np.zeros(n, bool)
            r["res"][i] += [int((rmask & rb & ra).sum()), int((rmask & rb & ~ra).sum()),
                            int((rmask & ~rb & ra).sum()), int((rmask & ~rb & ~ra).sum())]
            if np.isfinite(v):
                r["res_indep"][i] += int(np.count_nonzero(((a > 0) ^ (b > 0)) & (np.abs(a) > v)))
            r["res_notclip"][i] += int((rmask & ~clip).sum())
            r["res_r5_not_lc"][i] += int((rmask & (s5 != sa)).sum())
            r["bins_clip"][i] += np.bincount(idx[clip], minlength=4)[:4]
            r["rat_clip"][i] += ratio_hist(lr[clip])
            for k in range(4):
                r["rat_clip_q"][i][k] += ratio_hist(lr[clip & (idx == k)])
        for key, x in streams_of(a, b).items():
            ax = np.abs(x)
            conf = sig(ax)
            corr = ((x > 0) == bit)
            for scope, edges in (("unit", e_unit[key]), ("pool", e_pool[key])):
                bi = np.searchsorted(edges, ax, side="right")
                m = len(edges) + 1
                c = cal.setdefault((scope, key), [np.zeros(m, np.int64), np.zeros(m), np.zeros(m, np.int64)])
                c[0] += np.bincount(bi, minlength=m)
                c[1] += np.bincount(bi, weights=conf, minlength=m)
                c[2] += np.bincount(bi, weights=corr, minlength=m).astype(np.int64)
            fb = np.searchsorted(EDGES, ax, side="right") - 1
            r[f"rel_n_{key}"] += np.bincount(fb, minlength=NFINE)
            r[f"rel_c_{key}"] += np.bincount(fb, weights=corr, minlength=NFINE).astype(np.int64)
            r[f"rel_L_{key}"] += np.bincount(fb, weights=ax, minlength=NFINE)
    r["cal"] = cal
    return (seed, cond, t), r


def ece_of(c):
    cnt, sconf, scorr = c
    N = cnt.sum()
    ece, mce, rows = 0.0, 0.0, []
    for k in range(len(cnt)):
        if cnt[k] == 0: continue
        mc, acc = sconf[k] / cnt[k], scorr[k] / cnt[k]
        g = abs(mc - acc)
        ece += cnt[k] / N * g; mce = max(mce, g)
        rows.append((k, int(cnt[k]), mc, acc))
    return ece, mce, rows


def reliability(n_f, c_f, L_f):
    """8b from fine-bin sums: coarse table, slope, log-odds at targets, measurable range."""
    N = np.bincount(CMAP, weights=n_f, minlength=NCOARSE)
    NC = np.bincount(CMAP, weights=c_f, minlength=NCOARSE)
    SL = np.bincount(CMAP, weights=L_f, minlength=NCOARSE)
    err = N - NC
    table = []
    for k in range(NCOARSE):
        if N[k] == 0: continue
        table.append((k, C_LO[k], C_HI[k], int(N[k]), int(NC[k]), SL[k] / N[k],
                      math.log((NC[k] + 0.5) / (err[k] + 0.5))))
    ok = err >= MIN_ERR
    out = dict(slope=float("nan"), xmin=float("nan"), xmax=float("nan"), nbins=int(ok.sum()),
               maxbin_hi=float("nan"))
    for L in LTARGETS: out[f"lo{L}"] = float("nan")
    if ok.sum() >= 2:
        x = SL[ok] / N[ok]
        y = np.log((NC[ok] + 0.5) / (err[ok] + 0.5))
        w = 1.0 / (1.0 / (NC[ok] + 0.5) + 1.0 / (err[ok] + 0.5))
        out.update(slope=float((w * x * y).sum() / (w * x * x).sum()), xmin=float(x.min()),
                   xmax=float(x.max()), maxbin_hi=float(C_HI[np.nonzero(ok)[0].max()]))
        o = np.argsort(x)
        for L in LTARGETS:
            if x.min() <= L <= x.max():
                out[f"lo{L}"] = float(np.interp(L, x[o], y[o]))
    return out, table


def add(acc, r):
    for k, v in r.items():
        if k == "cal":
            for kk, c in v.items():
                if kk[0] != "pool": continue
                if kk not in acc.setdefault("cal", {}):
                    acc["cal"][kk] = [x.copy() for x in c]
                else:
                    for x, y in zip(acc["cal"][kk], c): x += y
        elif isinstance(v, np.ndarray):
            acc[k] = acc[k] + v if k in acc else v.copy()
        else:
            acc[k] = acc.get(k, 0) + v
    return acc


def rows_of(seed, cond, snr, r, hd):
    rq = hist_quantiles(hd, EDGES, RQ)
    out = []
    n = r["n"]
    cal = {k[1]: ece_of(c) for k, c in r["cal"].items()}
    rel = {k: reliability(r[f"rel_n_{k}"], r[f"rel_c_{k}"], r[f"rel_L_{k}"])[0] for k in STREAMS}
    for i, (name, v) in enumerate(DMAX):
        nc = int(r["n_clip"][i])
        f = (lambda x: x / nc) if nc else (lambda x: 0.0)
        j, rs = r["j"][i], r["res"][i]
        r5key = "Lc" if v == 0 else ("LR3" if not np.isfinite(v) else f"LR5_{name}")
        rq_all = ratio_quantiles(r["rat_all"], RATQ)
        rq_clip = ratio_quantiles(r["rat_clip"][i], RATQ)
        nres = int(rs.sum())
        row = dict(seed=seed, condition=cond, snr=snr, delta_max=name, delta_max_value=v,
                   n_bits=n, clip_fraction=nc / n, clip_upper=int(r["n_up"][i]) / n,
                   clip_lower=int(r["n_lo"][i]) / n,
                   **{f"resid_{nm}": x for nm, x in zip(RQN, rq)},
                   signdis_all=r["sd_all"] / n, signdis_clipped=f(int(r["sd_clip"][i])),
                   clip_r3right_lcright=f(int(j[0])), clip_r3right_lcwrong=f(int(j[1])),
                   clip_r3wrong_lcright=f(int(j[2])), clip_r3wrong_lcwrong=f(int(j[3])),
                   clip_r3_right=f(int(j[0] + j[1])), clip_r3_wrong=f(int(j[2] + j[3])),
                   clip_lc_right=f(int(j[0] + j[2])), clip_lc_wrong=f(int(j[1] + j[3])),
                   res_n=nres, res_share_of_clipped=f(nres),
                   res_reversed_correct=int(rs[1]), res_repaired=int(rs[2]),
                   res_both_right=int(rs[0]), res_both_wrong=int(rs[3]),
                   res_repaired_over_reversed=(int(rs[2]) / int(rs[1])) if rs[1] else float("nan"),
                   res_reversed_share_of_clipped=f(int(rs[1])), res_repaired_share_of_clipped=f(int(rs[2])),
                   clip_r5_reverses_r3=f(int(r["rev53"][i])), clip_r5_differs_lc=f(int(r["dif5c"][i])),
                   clip_samesign_trim=f(int(r["same_clip"][i])),
                   clip_signdiff=f(nc - int(r["same_clip"][i])),
                   **{f"ratio_all_p{int(q * 100)}": x for q, x in zip(RATQ, rq_all)},
                   **{f"ratio_clip_p{int(q * 100)}": x for q, x in zip(RATQ, rq_clip)},
                   **{f"ratio_all_p50_q{k + 1}": ratio_quantiles(r["rat_all_q"][k], [0.5])[0] for k in range(4)},
                   **{f"ratio_clip_p50_q{k + 1}": ratio_quantiles(r["rat_clip_q"][i][k], [0.5])[0] for k in range(4)},
                   ber_lc=r["err_lc"] / n, ber_r3=r["err_r3"] / n, ber_r5=int(r["err_r5"][i]) / n,
                   ber_lc_clipped=f(int(j[1] + j[3])), ber_r3_clipped=f(int(j[2] + j[3])),
                   ber_r5_clipped=f(int(r["err_r5_clip"][i])),
                   ece_lc=cal["Lc"][0], ece_r3=cal["LR3"][0], ece_r5=cal[r5key][0],
                   mce_lc=cal["Lc"][1], mce_r3=cal["LR3"][1], mce_r5=cal[r5key][1],
                   **{f"conf_bin{k + 1}": int(r["bins_clip"][i][k]) / max(int(r["bins_tot"][k]), 1) for k in range(4)},
                   mask_mismatch=int(r["mismatch"][i]),
                   res_indep=int(r["res_indep"][i]), res_not_clipped=int(r["res_notclip"][i]),
                   res_r5_not_lc_sign=int(r["res_r5_not_lc"][i]))
        for tag, key in (("lc", "Lc"), ("r3", "LR3"), ("r5", r5key)):
            rr = rel[key]
            row[f"slope_{tag}"] = rr["slope"]
            row[f"slope_{tag}_Lmin"] = rr["xmin"]; row[f"slope_{tag}_Lmax"] = rr["xmax"]
            row[f"slope_{tag}_nbins"] = rr["nbins"]; row[f"maxbin100_{tag}_Lhi"] = rr["maxbin_hi"]
            for L in LTARGETS:
                row[f"logodds_{tag}_at{L}"] = rr[f"lo{L}"]
        out.append(row)
    return out


def synthetic_gate(hist_lc):
    rng = np.random.default_rng(SYN_SEED)
    p = hist_lc / hist_lc.sum()
    k = rng.choice(len(hist_lc), size=SYN_N, p=p)
    lo, hi = EDGES[k], EDGES[k + 1]
    lo = np.where(lo <= 0, 1e-7, lo); hi = np.where(np.isinf(hi), lo * 1.0001, hi)
    mag = np.exp(np.log(lo) + rng.random(SYN_N) * (np.log(hi) - np.log(lo)))
    L = mag * np.where(rng.random(SYN_N) < 0.5, -1.0, 1.0)
    bit = rng.random(SYN_N) < sig(L)
    res = {}
    for name, X in (("calibrated", L), ("scaled_x100", 100.0 * L)):
        ax = np.abs(X)
        corr = ((X > 0) == bit)
        e = cal_edges(np.histogram(ax, bins=EDGES)[0])
        bi = np.searchsorted(e, ax, side="right"); m = len(e) + 1
        c = [np.bincount(bi, minlength=m), np.bincount(bi, weights=sig(ax), minlength=m),
             np.bincount(bi, weights=corr, minlength=m)]
        ece, mce, rows = ece_of(c)
        fb = np.searchsorted(EDGES, ax, side="right") - 1
        rel, _ = reliability(np.bincount(fb, minlength=NFINE).astype(np.float64),
                             np.bincount(fb, weights=corr, minlength=NFINE),
                             np.bincount(fb, weights=ax, minlength=NFINE))
        res[name] = dict(ece=ece, mce=mce, bins=len(rows), err=float(1 - corr.mean()), slope=rel["slope"],
                         Lmin=rel["xmin"], Lmax=rel["xmax"], nbins=rel["nbins"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gates", "full"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--workers", type=int, default=27)
    ap.add_argument("--out", default=None, help="output directory (default: the recorded results/run_b2_clipstats)")
    ap.add_argument("--joint", action="store_true", help="add seed 42's joint condition cdlD_irr20 (run J dumps)")
    args = ap.parse_args()
    global OUT
    if args.out: OUT = os.path.abspath(args.out)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    conds = RUNB_CONDS if args.mode == "gates" else CONDS
    conds_of = lambda s: conds + (["cdlD_irr20"] if args.joint and args.mode == "full" and s == 42 else [])
    seeds = [42] if args.mode == "gates" else args.seeds
    jobs, meta = [], {}
    for s in seeds:
        for c in conds_of(s):
            m, tags, snrs = tags_of(dump_dir(s, c))
            meta[(s, c)] = dict(zip(tags, snrs))
            jobs += [(s, c, t) for t in tags]
    print(f"[{time.strftime('%F %T')}] B2 {args.mode}: {len(jobs)} units, {args.workers} workers, "
          f"python {sys.executable}, numpy {np.__version__}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        p1 = dict(ex.map(pass1, jobs))
    print(f"[{time.strftime('%F %T')}] pass 1 done ({time.time() - t0:.0f} s)", flush=True)

    hpool, l2q, e_pool = {}, {}, {}
    for s in seeds:
        for c in conds_of(s):
            acc = {}
            for (ss, cc, t), h in p1.items():
                if (ss, cc) != (s, c): continue
                for k, v in h.items(): acc[k] = acc[k] + v if k in acc else v.copy()
            hpool[(s, c)] = acc
            l2q[(s, c)] = hist_quantiles(acc["Lc"], EDGES, [0.25, 0.5, 0.75])
            e_pool[(s, c)] = {k: cal_edges(v) for k, v in acc.items() if k != "d"}
    jobs2 = [((s, c, t), l2q[(s, c)], {k: cal_edges(v) for k, v in p1[(s, c, t)].items() if k != "d"},
              e_pool[(s, c)]) for (s, c, t) in jobs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        p2 = dict(ex.map(pass2, jobs2))
    print(f"[{time.strftime('%F %T')}] pass 2 done ({time.time() - t0:.0f} s)", flush=True)

    pooled_rows, snr_rows, cal_rows = [], [], []
    def cal_out(s, c, snr, rr, scope):
        for (sc, key), cc_ in rr["cal"].items():
            if sc != scope: continue
            for (k, cnt, mc, ac) in ece_of(cc_)[2]:
                cal_rows.append(dict(seed=s, condition=c, snr=snr, form="8a", stream=key, bin=k, lo=float("nan"),
                                     hi=float("nan"), n=cnt, n_correct=float("nan"), mean_stated=mc, empirical=ac))
        for key in STREAMS:
            _, table = reliability(rr[f"rel_n_{key}"], rr[f"rel_c_{key}"], rr[f"rel_L_{key}"])
            for (k, lo, hi, nn, ncc, mL, lo_) in table:
                cal_rows.append(dict(seed=s, condition=c, snr=snr, form="8b", stream=key, bin=k, lo=lo, hi=hi,
                                     n=nn, n_correct=ncc, mean_stated=mL, empirical=lo_))
    for s in seeds:
        for c in conds_of(s):
            acc = {}
            for (ss, cc, t), r in p2.items():
                if (ss, cc) != (s, c): continue
                add(acc, r)
                ru = dict(r); ru["cal"] = {k: v for k, v in r["cal"].items() if k[0] == "unit"}
                snr_rows += rows_of(s, c, meta[(s, c)][t], ru, p1[(s, c, t)]["d"])
                cal_out(s, c, meta[(s, c)][t], ru, "unit")
            pooled_rows += rows_of(s, c, "pooled", acc, hpool[(s, c)]["d"])
            cal_out(s, c, "pooled", acc, "pool")

    # ---------------- gates (order: G5, G1, G2, G3, G4) ----------------
    runb = {(r["condition"], r["delta_max"]): r for r in csv.DictReader(open(RUNB))}
    P = {(r["condition"], r["delta_max"]): r for r in pooled_rows if r["seed"] == 42}
    gates = []
    def g(gate, scope, crit, ok, detail=""):
        gates.append((gate, scope, crit, "PASS" if ok else "FAIL", detail))
    g("B2-G5", "environment", "numpy == 1.26.4 under the paper3 env python",
      np.__version__ == "1.26.4" and "/envs/paper3/" in sys.executable, f"numpy {np.__version__}; {sys.executable}")
    x = P[("cdlC_irr25", "ln4")]; rb = runb[("cdlC_irr25", "ln4")]
    for col in ("clip_fraction", "resid_p50"):
        g("B2-G1", "cdlC_irr25/ln4", f"{col} == run B {rb[col]}", fmt(x[col]) == rb[col], fmt(x[col]))
    for mine, theirs in (("clip_r3_wrong", "wrong_direction_share"), ("clip_r3_right", "right_sign_oversized_share"),
                         ("clip_lc_right", "l2_sign_correct_on_clipped")):
        g("B2-G1", "cdlC_irr25/ln4", f"{mine} == run B {theirs} {rb[theirs]}", fmt(x[mine]) == rb[theirs], fmt(x[mine]))
    x16 = P[("cdlC_irr25", "ln16")]
    g("B2-G1", "cdlC_irr25/ln16", f"clip_fraction == run B {runb[('cdlC_irr25', 'ln16')]['clip_fraction']}",
      fmt(x16["clip_fraction"]) == runb[("cdlC_irr25", "ln16")]["clip_fraction"], fmt(x16["clip_fraction"]))
    for c in RUNB_CONDS:
        g("B2-G1", f"{c}/ln4", f"wrong-direction share (R3 wrong) == run B {runb[(c, 'ln4')]['wrong_direction_share']}",
          fmt(P[(c, "ln4")]["clip_r3_wrong"]) == runb[(c, "ln4")]["wrong_direction_share"], fmt(P[(c, "ln4")]["clip_r3_wrong"]))
    same = tot = 0
    mism = []
    colmap = dict(clip_fraction="clip_fraction", clip_upper="clip_upper", clip_lower="clip_lower",
                  resid_p50="resid_p50", resid_p90="resid_p90", resid_p99="resid_p99", resid_p999="resid_p999",
                  clip_r3_wrong="wrong_direction_share", clip_r3_right="right_sign_oversized_share",
                  clip_lc_right="l2_sign_correct_on_clipped", conf_bin1="conf_bin1", conf_bin2="conf_bin2",
                  conf_bin3="conf_bin3", conf_bin4="conf_bin4", n_bits="n_bits")
    for c in RUNB_CONDS:
        for name, _ in DMAX:
            for mine, theirs in colmap.items():
                tot += 1
                ok = fmt(P[(c, name)][mine]) == runb[(c, name)][theirs]
                same += ok
                if not ok:
                    mism.append((c, name, theirs, fmt(P[(c, name)][mine]), runb[(c, name)][theirs]))
    with open(f"{OUT}/gate_g1full_mismatches.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["condition", "delta_max", "column", "b2", "run_b"]); w.writerows(mism)
    g("B2-G1", "3 conditions x 6 Delta_max x 15 columns", "every run-B value reproduced exactly", same == tot, f"{same}/{tot}")
    hb = np.load(RUNB_HIST_L2)
    g("B2-G1", "cdlC_irr25", "pooled |L_c| histogram == run B hist_absl2", np.array_equal(hb, hpool[(42, "cdlC_irr25")]["Lc"]))
    for r in pooled_rows:
        if np.isfinite(r["delta_max_value"]):
            g("B2-G2", f"seed{r['seed']}/{r['condition']}/{r['delta_max']}", "clip mask == (L_R5 != L_R3) mask",
              r["mask_mismatch"] == 0, str(r["mask_mismatch"]))
        else:
            g("B2-G2", f"seed{r['seed']}/{r['condition']}/unbounded", "clip fraction == 0 at inf", r["clip_fraction"] == 0.0)
    syn = synthetic_gate(hpool[(42, "cdlC_irr25")]["Lc"])
    cal_, x100 = syn["calibrated"], syn["scaled_x100"]
    det = lambda d: (f"ECE={d['ece']:.6g} MCE={d['mce']:.6g} slope={d['slope']:.6f} over |L| {d['Lmin']:.4g}-{d['Lmax']:.4g} "
                     f"in {d['nbins']} bins; err={d['err']:.6f}")
    g("B2-G3", "synthetic calibrated 1e7 (|L| ~ empirical |L_c| cdlC_irr25)", "8a ECE < 2e-3", cal_["ece"] < 2e-3, det(cal_))
    g("B2-G3", "synthetic calibrated 1e7", "8b slope in 1.00 +/- 0.02", abs(cal_["slope"] - 1.0) <= 0.02, f"slope={cal_['slope']:.6f}")
    g("B2-G3", "same stream x100", "8b slope in [0.005, 0.02]", 0.005 <= x100["slope"] <= 0.02, det(x100))
    ratio = x100["ece"] / cal_["ece"] if cal_["ece"] > 0 else float("inf")
    g("B2-G3", "same stream x100", "8a ECE >= 10 x calibrated ECE", ratio >= 10, f"ratio={ratio:.2f}")
    for r in pooled_rows:
        name, v = r["delta_max"], r["delta_max_value"]
        if not np.isfinite(v): continue
        sc = f"seed{r['seed']}/{r['condition']}/{name}"
        cells = (r["clip_r3right_lcright"] + r["clip_r3right_lcwrong"] + r["clip_r3wrong_lcright"]
                 + r["clip_r3wrong_lcwrong"])
        ok = abs(cells - 1.0) < 1e-12
        if r["seed"] == 42 and r["condition"] in RUNB_CONDS:
            rb = runb[(r["condition"], name)]
            ok = ok and (fmt(r["clip_r3_right"]) == rb["right_sign_oversized_share"]
                         and fmt(r["clip_r3_wrong"]) == rb["wrong_direction_share"]
                         and fmt(r["clip_lc_right"]) == rb["l2_sign_correct_on_clipped"])
        g("B2-G4", sc, "2x2 cells sum to 1; marginals == run B (run-B conditions)", ok)
        res_sum = r["res_reversed_correct"] + r["res_repaired"] + r["res_both_right"] + r["res_both_wrong"]
        g("B2-G4", sc, "restricted 2x2 sums to the sign-disagreeing |L_c| > Delta_max count", res_sum == r["res_indep"],
          f"{res_sum} vs {r['res_indep']}")
        g("B2-G4", sc, "restricted set: all clipped, L_R5 has the classical sign, off-diagonal cells 0",
          r["res_not_clipped"] == 0 and r["res_r5_not_lc_sign"] == 0 and r["res_both_right"] == 0 and r["res_both_wrong"] == 0,
          f"not_clipped={r['res_not_clipped']} r5_not_lc={r['res_r5_not_lc_sign']} both_right={r['res_both_right']} "
          f"both_wrong={r['res_both_wrong']}")

    with open(f"{OUT}/gates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gate", "scope", "criterion", "result", "detail"]); w.writerows(gates)
    bad = [x for x in gates if x[3] != "PASS"]
    print(f"\n=== RUN B2 GATES ({args.mode}) ===")
    for x in gates:
        if x[3] != "PASS" or x[0] in ("B2-G5", "B2-G3") or "every run-B value" in x[2]:
            print(f"  {x[0]:<6} {x[1][:58]:<58} {x[2][:52]:<52} {x[3]}  {x[4]}")
    by = {}
    for x in gates: by.setdefault(x[0], [0, 0]); by[x[0]][0] += x[3] == "PASS"; by[x[0]][1] += 1
    print("  " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by.items())))
    if args.mode == "full" and not bad:
        for name, rows in (("clipstats_summary.csv", pooled_rows), ("clipstats_per_snr.csv", snr_rows),
                           ("calibration_tables.csv", cal_rows)):
            with open(f"{OUT}/{name}", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("outputs written: clipstats_summary.csv clipstats_per_snr.csv calibration_tables.csv gates.csv")
    elif args.mode == "full":
        print("NOT WRITING outputs: a gate failed (see gates.csv)")
    print(f"[{time.strftime('%F %T')}] B2 {args.mode} finished in {time.time() - t0:.0f} s; "
          f"{'ALL GATES PASS' if not bad else f'{len(bad)} GATE FAILURE(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
