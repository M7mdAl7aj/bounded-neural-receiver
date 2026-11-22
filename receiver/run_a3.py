"""run_a3.py - run A3, three-seed fusion.

Run A's pipeline parameterised by seed and nothing else: the recorded scripts are
loaded UNCHANGED and only their path constants are pointed at one seed's dumps.

  offline   run_a_offline.main(): LLR-level endpoint gates A-G1..A-G4 (35 checks)
            and the F2/F3 grids from THIS seed's in-distribution dump.
  ag0       ag0_cpu_runner.run_shard(): the R5 ladder (6 Delta_max) decoded on CPU
            from the dumps, one shard per (condition, Delta_max).
  fusion    run_a_cpu.main(--shard): one shard per (condition, method, param),
            100 shards (5 conditions x 20 settings).
  summary   crossings with ra4_offline_ladder.crossing_10pct (run A's detector)
            and the gates:
              A3-G1..G4  every row of this seed's endpoint_gates.csv: 0.0e+00.
              A3-G5      F1 lambda = 0 / 1 crossings equal this seed's run-C
                         L2-real / R3 crossings (results/run_c_ci/ci_summary.csv,
                         4 dp, "unreachable" both or neither); the ladder shards
                         also match run C's online CSVs in integer error counts
                         (ag0_cpu_runner.aggregate's A-G0 comparison).
Outputs under results/run_a3/seed<S>/ (the recorded results/run_a_fusion and
results/run_c_ci/ag0 trees are never written). CPU only: CUDA_VISIBLE_DEVICES=""
is required (run_a_cpu.py asserts it at import).

  python run_a3.py --seed 42 --mode offline
  python run_a3.py --seed 42 --mode ag0 --condition cdlC_irr25 --dmax ln4
  python run_a3.py --seed 42 --mode fusion --condition cdlC_irr25 --method F1 --param 0.5
  python run_a3.py --seed 42 --mode summary
"""
import argparse, csv, importlib.util, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
CONDS = ["cdlC_irr25", "irr20", "irr35", "cdlD", "cdlE"]
DMAX = ["0", "ln2", "ln4", "ln8", "ln16", "unbounded"]
SPECS = ["F1:0", "F1:0.25", "F1:0.5", "F1:0.75", "F1:1",
         "F2:0", "F2:p50", "F2:p75", "F2:p90", "F2:inf",
         "F3:inf", "F3:p90", "F3:p75", "F3:p50", "F3:0",
         "F4:0.6931471805599453", "F4:1.3862943611198906",
         "F4:2.0794415416798357", "F4:2.772588722239781", "F4:1e6"]


def load(name, fname):
    sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


def paths(seed):
    out = f"results/run_a3/seed{seed}"
    return dict(DUMPS=f"results/run_c_ci/dumps/seed{seed}", ONLINE=f"results/run_c_ci/seed{seed}",
                OUT=out, AG0=f"{out}/ag0_shards", SHARDS=f"{out}/shards")


def mode_offline(P):
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        sys.exit("refused: CUDA_VISIBLE_DEVICES must be the empty string")
    rao = load("run_a_offline", "run_a_offline.py")
    rao.DUMPS, rao.OUT = P["DUMPS"], P["OUT"]
    return rao.main()


def mode_ag0(P, cond, dname):
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        sys.exit("refused: CUDA_VISIBLE_DEVICES must be the empty string")
    ag = load("ag0_cpu_runner", "ag0_cpu_runner.py")
    ag.DUMPS, ag.ONLINE, ag.SHARDS = P["DUMPS"], P["ONLINE"], P["AG0"]
    ag.run_shard(cond, dname, 8)
    return 0


def mode_fusion(P, cond, method, param):
    rac = load("run_a_cpu", "run_a_cpu.py")           # asserts CUDA_VISIBLE_DEVICES == ""
    rac.DUMPS, rac.OUT, rac.SHARDS = P["DUMPS"], P["OUT"], P["SHARDS"]
    sys.argv = ["run_a_cpu.py", "--shard", "--condition", cond, "--method", method, "--param", param]
    return rac.main()


def crossing(snr, errors, n):
    b = [e / n for e in errors]
    for i in range(1, len(snr)):                       # ra4_offline_ladder.crossing_10pct, verbatim
        if b[i - 1] >= 0.1 > b[i] and b[i - 1] > 0 and b[i] > 0:
            l1, l2 = math.log10(b[i - 1]), math.log10(b[i])
            return snr[i - 1] + (math.log10(0.1) - l1) * (snr[i] - snr[i - 1]) / (l2 - l1)
    return None


def fmt(c, dp):
    return "unreachable" if c is None else f"{c:.{dp}f}"


def mode_summary(P, seed):
    rows, verdicts = [], []
    def jl(p): return json.load(open(p))
    ag0, fus = {}, {}
    for cond in CONDS:
        for d in DMAX:
            f = f"{P['AG0']}/{cond}__{d}.json"
            if not os.path.exists(f): sys.exit(f"missing ladder shard {f}")
            sh = jl(f); ag0[(cond, d)] = (crossing(sh["snr_db"], sh["errors"], sh["n_blocks"]), sh)
        for spec in SPECS:
            m, p = spec.split(":")
            f = f"{P['SHARDS']}/{cond}__{m}__{p}.json"
            if not os.path.exists(f): sys.exit(f"missing fusion shard {f}")
            sh = jl(f); fus[(cond, m, p)] = (crossing(sh["snr_db"], sh["errors"], sh["n_blocks"]), sh)
    with open(f"{P['OUT']}/a3_summary.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["seed", "condition", "kind", "param", "crossing_db", "min_bler", "errors"])
        for (cond, d), (c, sh) in ag0.items():
            w.writerow([seed, cond, "R5", d, fmt(c, 4), min(sh["errors"]) / sh["n_blocks"], ";".join(map(str, sh["errors"]))])
        for (cond, m, p), (c, sh) in fus.items():
            w.writerow([seed, cond, m, p, fmt(c, 4), min(sh["errors"]) / sh["n_blocks"], ";".join(map(str, sh["errors"]))])

    # A3-G1..G4
    eg = list(csv.DictReader(open(f"{P['OUT']}/endpoint_gates.csv")))
    bad = [r for r in eg if float(r["max_abs_diff"]) != 0.0 or r["result"] != "PASS"]
    verdicts.append(("A3-G1..G4", not bad and len(eg) == 35, f"{len(eg)} checks, {len(bad)} nonzero"))

    # A3-G5: F1 endpoints vs run-C per-seed crossings, and ladder vs online CSVs (integer counts)
    ci = {(r["condition"], r["delta_max"]): r["crossing_db"] for r in csv.DictReader(open("results/run_c_ci/ci_summary.csv"))
          if r["seed"] == str(seed)}
    g5 = []
    for cond in CONDS:
        for lam, d in (("0", "0"), ("1", "unbounded")):
            got = fmt(fus[(cond, "F1", lam)][0], 4); want = ci.get((cond, d), "MISSING")
            want_n = "unreachable" if not re.match(r"^-?[0-9.]+$", want or "") else want
            g5.append((cond, lam, got, want, got == want_n))
    mism = 0
    for cond in CONDS:
        meta = jl(f"{P['DUMPS']}/{cond}/index.json"); irr = int(meta["eval_irr_db"]); N = meta["n_blocks"]
        for d in DMAX:
            on = list(csv.DictReader(open(f"{P['ONLINE']}/{cond}/r5inf_bler_dmax{d}_irr{irr}.csv")))
            errs = ag0[(cond, d)][1]["errors"]
            mism += sum(1 for i, r in enumerate(on) if round(float(r["bler_mean"]) * N) != errs[i])
    ok5 = all(x[4] for x in g5)
    verdicts.append(("A3-G5", ok5, "; ".join(f"{c} l={l}: {g} vs run-C {w}" for c, l, g, w, ok in g5 if not ok) or
                     f"10/10 endpoint crossings equal run C (4 dp); ladder vs online CSVs: {mism} integer mismatches"))
    verdicts.append(("A3-G5b (ladder integer counts, report)", mism == 0, f"{mism} integer mismatches over 5 x 6 x 9 points"))

    with open(f"{P['OUT']}/a3_gates.txt", "w") as fo:
        for name, ok, det in verdicts:
            line = f"{name}: {'PASS' if ok else 'FAIL'} - {det}"
            print(line); fo.write(line + "\n")
    hard = [v for v in verdicts if not v[0].endswith("(ladder integer counts, report)")]
    return 0 if all(ok for _, ok, _ in hard) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["offline", "ag0", "fusion", "summary"], required=True)
    ap.add_argument("--condition"); ap.add_argument("--dmax"); ap.add_argument("--method"); ap.add_argument("--param")
    a = ap.parse_args()
    P = paths(a.seed)
    if not os.path.isdir(P["DUMPS"]): sys.exit(f"no dumps for seed {a.seed}")
    os.makedirs(P["OUT"], exist_ok=True)
    if a.mode == "offline": return mode_offline(P)
    if a.mode == "ag0": return mode_ag0(P, a.condition, a.dmax)
    if a.mode == "fusion": return mode_fusion(P, a.condition, a.method, a.param)
    return mode_summary(P, a.seed)


if __name__ == "__main__":
    sys.exit(main())
