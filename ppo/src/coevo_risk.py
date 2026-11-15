# Risk metrics by segment for frozen BTC models (no retraining): Sharpe, Sortino, max drawdown, return.
#   python coevo_risk.py <node>  ->  <node>/BTCUSDT/risk.csv
# Passive baselines: buy_hold (spot), long (the agent's instrument held long, agent costs + funding), ma_crossover.
import sys, os, json, multiprocessing as mp, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import coevo_eval as E
METRICS = ("sharpe", "sortino", "max_drawdown_pct", "total_return_pct")
node = sys.argv[1]
root = f"{E.OUT_BASE}/{node}/BTCUSDT"
E.CFG = json.load(open(f"{root}/contract.json"))["config"]
seeds = E.CFG["seeds"]
assert E.CFG.get("score") == "vote"


def job(j):
    k, path = j
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace as NS
    from stable_baselines3 import PPO
    from trading_env import TradingEnv
    from metrics import summarize
    _, test = E.DATA[k]
    segs = [(s, s + E.SEG) for s in range(0, len(test) - E.SEG + 1, E.SEG)]
    if path == "long":
        model = NS(predict=lambda obs: (1, None))
    else:
        d = os.path.join(os.path.dirname(path), "final_traders")
        model = E.SoftVote([PPO.load(os.path.join(d, f), device="cpu") for f in sorted(os.listdir(d)) if f.endswith(".zip")])
    eng = E.CFG.get("engine", {})
    out = []
    for i in range(len(segs)):
        env = TradingEnv(df=test, feature_columns=E.FEATS, segments=segs, initial_balance=10000.0,
                         trading_fee=E.AGENT_FEE, slippage=E.AGENT_SLIP, trade_penalty=0.0,
                         funding_col="funding_paid", decision_every=eng.get("decision_every", 1),
                         start_long=eng.get("start_long", False), target_position=eng.get("target_position", False))
        obs, _ = env.reset(seed=0, options={"segment": i})
        v, done = [10000.0], False
        while not done:
            obs, _, done, _, info = env.step(model.predict(obs)[0])
            v.extend(info["bar_values"])
        m = summarize(v)
        out.append([m[x] for x in METRICS])
    return j, out


E.DATA, E.FEATS, _ = E.prepare(E.SYM, E.CFG)
jobs = [(k, f"{root}/fold{k}/{a}_seed{s}/best_trader.zip") for a in E.ARMS for k in sorted(E.DATA) for s in seeds]
jobs += [(k, "long") for k in sorted(E.DATA)]
with mp.get_context("fork").Pool(int(os.environ.get("W", 11))) as p:
    res = dict(p.map(job, jobs))
from baselines import buy_and_hold, ma_crossover
from metrics import summarize
rows = []
for k in sorted(E.DATA):
    test = E.DATA[k][1]
    for i, s in enumerate(range(0, len(test) - E.SEG + 1, E.SEG)):
        seg = test.iloc[s:s + E.SEG].reset_index(drop=True)
        r = {"fold": k, "seg": i, "move": float(np.log(seg.close.iloc[-1] / seg.close.iloc[0]))}
        base = {"buy_hold": summarize(buy_and_hold(seg, 10000.0, E.SPOT_FEE, E.SPOT_SLIP)),
                "ma_crossover": summarize(ma_crossover(seg, 10000.0, E.AGENT_FEE, slippage=E.AGENT_SLIP))}
        vals = {"long": res[(k, "long")][i]}
        for a in E.ARMS:
            vals[a] = np.mean([res[(k, f"{root}/fold{k}/{a}_seed{sd}/best_trader.zip")][i] for sd in seeds], axis=0)
        for n, m in base.items():
            vals[n] = [m[x] for x in METRICS]
        for n, v in vals.items():
            for x, y in zip(METRICS, v):
                r[f"{n}_{x}"] = float(y)
        rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(f"{root}/risk.csv", index=False)
old = pd.read_csv(f"{root}/per_segment.csv")   # self-check: the rescored Sharpe must reproduce the harness
for a in list(E.ARMS) + ["buy_hold"]:
    assert np.allclose(d[f"{a}_sharpe"], old[a], atol=1e-6), a
print(node, "ok")
