"""Exploratory rescore (not the frozen harness): each arm on each fold is ONE soft-vote policy over the final
traders of ALL seeds, instead of the mean of per-seed scores. Dev folds only. Writes
runs/coevo7_ens/<node>/BTCUSDT/{per_segment.csv, contract.json} so coevo_v9.py can score it (BASE patched).

    python ens_rescore.py v9g-signed-progress v9g-signed-progress-seeds48-53 ...
"""
import json, multiprocessing as mp, os, shutil, sys
import numpy as np
import pandas as pd
import coevo_eval as ce

ENS = os.path.join(os.path.dirname(ce.OUT_BASE), "coevo7_ens")


def eval_ens(job):
    fold, dirs, mult = job
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from stable_baselines3 import PPO
    from coevo.trader_population import evaluate_on_segments
    models = [PPO.load(os.path.join(d, "final_traders", f), device="cpu")
              for d in dirs for f in sorted(os.listdir(os.path.join(d, "final_traders"))) if f.endswith(".zip")]
    _, test = ce.DATA[fold]
    segs = [(s, s + ce.SEG) for s in range(0, len(test) - ce.SEG + 1, ce.SEG)]
    e = ce.CFG.get("engine", {})
    kw = {"initial_balance": 10000.0, "trading_fee": ce.AGENT_FEE * mult, "slippage": ce.AGENT_SLIP * mult,
          "trade_penalty": 0.0, "funding_col": "funding_paid", "decision_every": e.get("decision_every", 1),
          "start_long": e.get("start_long", False), "target_position": e.get("target_position", False)}
    _, rows = evaluate_on_segments(SimpleNamespace(model=ce.SoftVote(models)), test, ce.FEATS, segs, kw)
    return job[0], tuple(dirs), mult, [(r["sharpe"], r["exposure"]) for r in rows]


def rescore(node, workers):
    src = os.path.join(ce.OUT_BASE, node, ce.SYM)
    out = os.path.join(ENS, node, ce.SYM)
    if os.path.exists(os.path.join(out, "per_segment.csv")):
        return
    contract = json.load(open(os.path.join(src, "contract.json")))
    ce.CFG = contract["config"]
    ce.DATA, ce.FEATS, wins = ce.prepare(ce.SYM, ce.CFG)
    assert all(w["test_start"] >= 40000 for w in wins)          # dev folds only, never E0-E2
    seeds, folds = ce.CFG["seeds"], [w["fold"] for w in wins]
    jobs = [(k, [os.path.join(src, f"fold{k}", f"{a}_seed{s}") for s in seeds], m)
            for a in ce.ARMS for k in folds for m in (1.0, 2.0)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = {(k, d, m): v for k, d, m, v in pool.imap_unordered(eval_ens, jobs)}
    old = pd.read_csv(os.path.join(src, "per_segment.csv"))           # baselines + move are unchanged
    seg = old[["fold", "seg", "move", "buy_hold", "ma_crossover", "buy_hold_2x", "ma_crossover_2x"]].copy()
    for a in ce.ARMS:
        for m, suf in ((1.0, ""), (2.0, "_2x")):
            col, ex = [], []
            for k, i in zip(seg.fold, seg.seg):
                v = res[(k, tuple(os.path.join(src, f"fold{k}", f"{a}_seed{s}") for s in seeds), m)][i]
                col.append(v[0]); ex.append(v[1])
            seg[a + suf] = col
            if m == 1.0:
                seg[a + "_expo"] = ex
    os.makedirs(out, exist_ok=True)
    seg.to_csv(os.path.join(out, "per_segment.csv"), index=False)
    shutil.copy(os.path.join(src, "contract.json"), out)
    print(node, {a: round(seg[a].mean(), 3) for a in ce.ARMS}, "(old", {a: round(old[a].mean(), 3) for a in ce.ARMS},
          ")", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        rescore(n, 11)
