"""Funding / open-interest probe (dev folds only, no RL, not a trial).

Q1: do derivatives features predict a >5% drawdown in the next 72 bars better than volatility (AUC 0.60)?
Q2: do extremes (funding, OI build-up, crowding) shift forward returns? (decile means, per fold)
Q3: does "go flat when predicted drawdown risk is high" beat buy & hold on the 48 dev segments?

    python deriv_probe.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from coin_data import fold_windows, load_coin
from metrics import summarize

SEG, FEE = 720, 0.0003

df = load_coin("BTCUSDT")
wins = fold_windows(df)
for w in wins:
    assert w["test_start"] >= 40000 or w["test_end"] <= 22000      # never the sealed E0-E2 bars

r = np.log(df.close).diff()
rv = lambda h: np.log(np.sqrt((r ** 2).rolling(h).sum()))
z = lambda s, n: (s - s.rolling(n, min_periods=n // 4).mean()) / s.rolling(n, min_periods=n // 4).std()

fund = df.funding_paid.where(df.funding_paid != 0).ffill()           # latest settled rate, causal
oi168 = df.oi_d1.rolling(168).sum()                                   # 7-day log change in open interest
D = pd.DataFrame({
    "fund": fund, "fund_3d": fund.rolling(72).mean(), "fund_z30": z(fund, 720),
    "oi_d24": df.oi_d24, "oi_d168": oi168, "oi_z30": z(oi168, 720),
    "ls_top": df.ls_top, "ls_all": df.ls_all, "taker_ls": df.taker_ls,
    "ls_all_z30": z(df.ls_all, 720), "taker_z": z(df.taker_ls.rolling(24).mean(), 720),
})
V = pd.DataFrame({f"rv{h}": rv(h) for h in (24, 168, 720)})
dd72 = df.close[::-1].rolling(72, min_periods=72).min()[::-1].shift(-1) / df.close - 1
y_dd = (dd72 < -0.05).astype(float).where(dd72.notna())
fwd72 = np.log(df.close.shift(-72) / df.close)
y_dir = (fwd72 > 0).astype(float).where(fwd72.notna())
sets = {"vol": V, "deriv": D, "vol+deriv": pd.concat([V, D], axis=1)}


def fitpred(X, y, tr, te):
    m = np.isfinite(y.iloc[tr].values)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    clf.fit(X.iloc[tr].values[m], y.iloc[tr].values[m])
    return clf.predict_proba(X.iloc[te].values)[:, 1]


rows, risk = [], {}
for w in wins:
    tr = np.arange(w["train_start"], w["train_end"] - 72)            # purge labels reaching into test
    te = np.arange(w["test_start"], w["test_end"] - 72)
    row = {"fold": w["fold"], "deriv_cover_train": D.iloc[tr].notna().all(axis=1).mean()}
    for name, X in sets.items():
        for tgt, y in (("dd", y_dd), ("dir", y_dir)):
            p = fitpred(X, y, tr, te)
            ok = np.isfinite(y.iloc[te].values)
            row[f"{tgt}_{name}"] = roc_auc_score(y.iloc[te].values[ok], p[ok])
            if tgt == "dd" and name == "vol+deriv":
                full = np.arange(w["test_start"] - 2160, w["test_end"])   # warm-up for the causal threshold
                s = np.full(len(df), np.nan)
                s[full] = fitpred(X, y, tr, full)
                risk[w["fold"]] = s
    rows.append(row)
res = pd.DataFrame(rows)
print("Q1: AUC per fold (dd = >5% drawdown in next 72 bars; dir = next-72-bar direction)")
print(res.round(3).to_string(index=False))
print("\nmean:\n", res.drop(columns="fold").mean().round(3).to_string())

print("\nQ2: mean next-72-bar log return (%) by decile of each feature, test rows of all folds pooled;"
      " 'folds_same_sign' = folds where top-minus-bottom decile has the pooled sign")
te_all = np.concatenate([np.arange(w["test_start"], w["test_end"] - 72) for w in wins])
fold_of = np.concatenate([np.full(w["test_end"] - 72 - w["test_start"], w["fold"]) for w in wins])
q2 = []
for c in D.columns:
    x, y = D[c].iloc[te_all].values, fwd72.iloc[te_all].values
    ok = np.isfinite(x) & np.isfinite(y)
    dec = pd.qcut(pd.Series(x[ok]).rank(method="first"), 10, labels=False).values
    m = pd.Series(y[ok]).groupby(dec).mean() * 100
    spread = m.iloc[-1] - m.iloc[0]
    per = []
    for f in range(6):
        k = ok & (fold_of == f)
        if k.sum() > 500:
            d = pd.qcut(pd.Series(x[k]).rank(method="first"), 10, labels=False).values
            g = pd.Series(y[k]).groupby(d).mean()
            per.append(np.sign(g.iloc[-1] - g.iloc[0]) == np.sign(spread))
    q2.append({"feature": c, "bottom": m.iloc[0], "top": m.iloc[-1], "top_minus_bottom": spread,
               "folds_same_sign": f"{sum(per)}/{len(per)}"})
print(pd.DataFrame(q2).round(3).to_string(index=False))


def seg_sharpe(expo, idx):
    ret = np.exp(r.iloc[idx].values) - 1
    e = np.r_[expo[0], expo[:-1]]
    pnl = e * ret - FEE * np.abs(np.diff(np.r_[0.0, e]))
    return summarize(10000 * np.cumprod(1 + pnl))["sharpe"], e.mean()


print("\nQ3: flat when predicted drawdown risk > its trailing-90-day quantile q (vol+deriv model), 48 dev segments")
tab = []
for q in (None, 0.7, 0.8, 0.9):
    rs = []
    for w in wins:
        s = pd.Series(risk[w["fold"]])
        thr = s.rolling(2160, min_periods=500).quantile(q).values if q else None
        for a in range(w["test_start"], w["test_end"] - SEG + 1, SEG):
            idx = np.arange(a, a + SEG)
            move = np.log(df.close.iloc[idx[-1]] / df.close.iloc[idx[0]])
            e = np.ones(SEG) if q is None else (s.values[idx] <= thr[idx]).astype(float)
            sh, ex = seg_sharpe(e, idx)
            rs.append({"regime": "bull" if move > 0.05 else ("bear" if move < -0.05 else "side"), "sharpe": sh, "expo": ex})
    t = pd.DataFrame(rs)
    g = t.groupby("regime").sharpe.mean()
    tab.append({"rule": "buy&hold" if q is None else f"flat above q{q}", "sharpe": t.sharpe.mean(),
                "expo": t.expo.mean(), "bear": g["bear"], "bull": g["bull"], "side": g["side"]})
print(pd.DataFrame(tab).round(3).to_string(index=False))
