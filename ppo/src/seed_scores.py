"""V10 rule (LEDGER "V10 PRE-REGISTRATION"): per-seed scores with the harness's own eval_job, then S1-S3.

    python seed_scores.py NODE [--workers 11]
Writes runs/coevo7/<node>/BTCUSDT/per_seed.csv. Dev folds only.
"""
import argparse, json, multiprocessing as mp, os
import numpy as np
import pandas as pd
import coevo_eval as ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node"); ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    src = os.path.join(ce.OUT_BASE, a.node, ce.SYM)
    out = os.path.join(src, "per_seed.csv")
    seg = pd.read_csv(os.path.join(src, "per_segment.csv"))
    if not os.path.exists(out):
        ce.CFG = json.load(open(os.path.join(src, "contract.json")))["config"]
        ce.DATA, ce.FEATS, wins = ce.prepare(ce.SYM, ce.CFG)
        assert all(w["test_start"] >= 40000 for w in wins)          # never E0-E2
        jobs = [(k, os.path.join(src, f"fold{k}", f"{arm}_seed{s}", "best_trader.zip"), 1.0)
                for arm in ce.ARMS for k in sorted(seg.fold.unique()) for s in ce.CFG["seeds"]]
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(ce.eval_job, jobs)
        rows = [{"arm": p.split(os.sep)[-2].rsplit("_seed", 1)[0], "seed": int(p.split(os.sep)[-2].rsplit("_seed", 1)[1]),
                 "fold": k, "seg": i, "sharpe": v[0], "expo": v[1]}
                for (k, p, _), vals in res for i, v in enumerate(vals)]
        pd.DataFrame(rows).to_csv(out, index=False)
    ps = pd.read_csv(out)
    for arm in ce.ARMS:                                              # must reproduce the harness exactly
        m = ps[ps.arm == arm].groupby(["fold", "seg"]).sharpe.mean().values
        assert np.allclose(m, seg[arm].values, atol=1e-9), arm
    per_seed = ps.groupby(["arm", "seed"]).sharpe.mean().unstack(0)
    wf = {x: seg.groupby("fold")[x].mean().min() for x in ("population", "coevolution")}
    sd = per_seed.std()
    r = {"node": a.node, "mean": {x: float(seg[x].mean()) for x in ce.ARMS},
         "per_seed_mean": per_seed.round(3).to_dict(), "seed_sd": sd.round(3).to_dict(), "worst_fold": wf,
         "S1": bool(wf["coevolution"] >= wf["population"]),
         "S2": bool(sd["coevolution"] < sd["population"]),
         "S3": bool(seg.coevolution.mean() - seg.population.mean() > -0.10)}
    r["PASS"] = r["S1"] and r["S2"] and r["S3"]
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
