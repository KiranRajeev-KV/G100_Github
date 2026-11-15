"""V9 rule (frozen in LEDGER "V9 PRE-REGISTRATION"): pool seed set A + B of one node, print P1/P2/P3 + PASS.

    python coevo_v9.py r21a-pure-alp r22a-pure-alp-seeds48-53            # dev
    python coevo_v9.py v9x v9x-seeds48-53 --confirm                     # E0-E2, ONCE
"""
import argparse, json
import numpy as np
import pandas as pd
from evaluate import paired_bootstrap
from regimes import label_regimes
from coin_data import SEALED_FROM

BASE, SEG = "../runs/coevo7", 720
ARMS = ["single", "population", "coevolution"]


def regime(move):
    """THE bear/bull definition used by the rule: the segment's own log move, +-0.05. This is the rule that
    reproduces runs/coevo7/btc_alp_pooled.csv's regime column exactly (see LEDGER "V9 STEP 0", NEEDS USER)."""
    return "bull" if move > 0.05 else ("bear" if move < -0.05 else "sideways")


def trailing_labels(seg, folds):
    """Reported only: majority trailing-720 label_regimes inside each segment (the pre-reg text's wording)."""
    df = pd.read_parquet("../data/features/btcusdt_features_ext2.parquet", columns=["open_time", "close"])
    df = df[df.open_time.astype("datetime64[ns]") < SEALED_FROM]
    lab = label_regimes(df, window=720, threshold=0.10)
    ts = {f["fold"]: f["test_start"] for f in folds}
    return [lab.iloc[ts[k] + i * SEG: ts[k] + (i + 1) * SEG].mode().iloc[0] for k, i in zip(seg.fold, seg.seg)]


def load(n, confirm):
    d = f"{BASE}/{n}_confirm/BTCUSDT" if confirm else f"{BASE}/{n}/BTCUSDT"
    return pd.read_csv(f"{d}/per_segment.csv"), json.load(open(f"{d}/contract.json"))["folds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node_a"); ap.add_argument("node_b")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--csv", default="", help="optional: write the pooled per-segment table here")
    a = ap.parse_args()
    (da, fa), (db, fb) = load(a.node_a, a.confirm), load(a.node_b, a.confirm)
    assert (da[["fold", "seg"]].values == db[["fold", "seg"]].values).all() and fa == fb
    seg = da.copy()
    for col in [x + s for x in ARMS for s in ("", "_2x", "_expo")]:
        seg[col] = (da[col].values + db[col].values) / 2
    seg["regime"] = seg.move.map(regime)
    seg["regime_trailing"] = trailing_labels(seg, fa)
    if a.csv:
        seg.to_csv(a.csv, index=False)
    c, p = seg.coevolution.values, seg.population.values
    bear = (seg.regime == "bear").values
    others = ["single", "population", "buy_hold", "ma_crossover"]
    fold_c, fold_p = seg.groupby("fold").coevolution.mean(), seg.groupby("fold").population.mean()
    r = {"nodes": [a.node_a, a.node_b], "confirm": a.confirm, "n_segments": len(seg), "n_bear": int(bear.sum()),
         **{x: float(seg[x].mean()) for x in ARMS + ["buy_hold", "ma_crossover"]},
         "set_gap_pop": {"A": float((da.coevolution - da.population).mean()),
                         "B": float((db.coevolution - db.population).mean())},
         "gaps": {o: float((c - seg[o].values).mean()) for o in others},
         "p": {o: paired_bootstrap(c, seg[o].values) for o in others},
         "bear_gap": float((c - p)[bear].mean()), "bear_p": paired_bootstrap(c[bear], p[bear]),
         "bear_trailing_reported": {"n": int((seg.regime_trailing == "bear").sum()),
                                    "gap": float((c - p)[(seg.regime_trailing == "bear").values].mean()),
                                    "p": paired_bootstrap(c[(seg.regime_trailing == "bear").values],
                                                          p[(seg.regime_trailing == "bear").values])},
         "regime_gap": {g: float((c - p)[(seg.regime == g).values].mean()) for g in ("bear", "bull", "sideways")},
         "worst_fold": {"coevolution": float(fold_c.min()), "population": float(fold_p.min())},
         "per_fold_gap_pop": (fold_c - fold_p).round(3).to_dict(),
         "coevolution_2x": float(seg.coevolution_2x.mean()), "buy_hold_2x": float(seg.buy_hold_2x.mean()),
         "exposure": {x: float(seg[x + "_expo"].mean()) for x in ARMS},
         "timing": {x: float(seg[x + "_expo"][seg.move > 0].mean() - seg[x + "_expo"][seg.move < 0].mean())
                    for x in ARMS}}
    r["P1"] = bool(all(r["gaps"][o] > 0 for o in others) and r["set_gap_pop"]["A"] > 0 and r["set_gap_pop"]["B"] > 0)
    r["P2"] = bool(r["bear_gap"] > 0 and r["bear_p"] <= 0.10
                   and r["worst_fold"]["coevolution"] >= r["worst_fold"]["population"])
    r["P3"] = bool(all(r["gaps"][o] > 0 and r["p"][o] <= 0.10 for o in ("single", "ma_crossover"))
                   and r["coevolution_2x"] > r["buy_hold_2x"])
    r["PASS"] = r["P1"] and r["P2"] and r["P3"]
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
