"""Volatility predictability probe (dev folds only, no RL, not a trial).

Q1: can next-24h / 72h realised vol be forecast better than persistence (and HAR)?
Q2: do features predict big drawdowns (next 72 bars < -5%)?  direction AUC as reference.
Q3: does vol-targeted exposure beat buy & hold on the 48 dev segments (mean segment Sharpe)?

    python vol_probe.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

from build_features import FEATURE_COLUMNS
from coin_data import FLOW, POS, fold_windows, load_coin
from metrics import summarize

SEG, FEE = 720, 0.0003            # agent fee + slippage per unit of turnover (funding ignored)

df = load_coin("BTCUSDT")
wins = fold_windows(df)
for w in wins:                    # dev tests only: never the sealed E0-E2 bars or the spent tail
    assert w["test_start"] >= 40000 or w["test_end"] <= 22000

r = np.log(df.close).diff()
rv = lambda h: np.log(np.sqrt((r ** 2).rolling(h).sum()))          # trailing log RV over h bars
fwd = lambda h: rv(h).shift(-h)                                     # next h bars
har = pd.DataFrame({f"rv{h}": rv(h) for h in (24, 168, 720)})
X = pd.concat([df[list(FEATURE_COLUMNS) + FLOW + POS], har], axis=1)
dd72 = (df.close[::-1].rolling(72, min_periods=72).min()[::-1].shift(-1) / df.close - 1)  # worst close in next 72
y_dd = (dd72 < -0.05).astype(float).where(dd72.notna())
y_dir = (df.close.shift(-24) > df.close).astype(float).where(df.close.shift(-24).notna())


def split(w, h):
    tr = np.arange(w["train_start"], w["train_end"] - h)            # purge labels that reach into test
    te = np.arange(w["test_start"], w["test_end"] - h)
    return tr, te


def fitpred(model, cols, y, tr, te):
    Xtr, Xte = X.iloc[tr][cols].values, X.iloc[te][cols].values
    if isinstance(model, Ridge):                                     # HGB handles NaN natively
        m = np.isfinite(y.iloc[tr].values) & np.isfinite(Xtr).all(axis=1)
        Xte = np.nan_to_num(Xte)
    else:
        m = np.isfinite(y.iloc[tr].values)
    model.fit(Xtr[m], y.iloc[tr].values[m])
    return model.predict(Xte) if not hasattr(model, "predict_proba") else model.predict_proba(Xte)[:, 1]


allc = list(X.columns)
rows, sigma = [], {}
for w in wins:
    f = w["fold"]
    for h in (24, 72):
        tr, te = split(w, h)
        y = fwd(h)
        yt = y.iloc[te].values
        ok = np.isfinite(yt)
        persist = rv(h).iloc[te].values                              # vol stays what it was
        harp = fitpred(Ridge(1.0), ["rv24", "rv168", "rv720"], y, tr, te)
        hgb = fitpred(HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05), allc, y, tr, te)
        rows.append({"fold": f, "h": h,
                     "r2_persist": r2_score(yt[ok], persist[ok]), "r2_har": r2_score(yt[ok], harp[ok]),
                     "r2_hgb_all": r2_score(yt[ok], hgb[ok]),
                     "corr_hgb": np.corrcoef(yt[ok], hgb[ok])[0, 1]})
        if h == 24:
            s = np.full(len(df), np.nan)
            s[te] = hgb
            sigma[f] = s
    tr, te = split(w, 72)
    for name, y in (("dd", y_dd), ("dir", y_dir)):
        p = fitpred(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05), allc, y, tr, te)
        yt = y.iloc[te].values
        ok = np.isfinite(yt)
        rows[-1][f"auc_{name}"] = roc_auc_score(yt[ok], p[ok])
        if name == "dd":
            rows[-1]["auc_dd_rv168_only"] = roc_auc_score(yt[ok], har.rv168.iloc[te].values[ok])
            rows[-1]["dd_base_rate"] = yt[ok].mean()

res = pd.DataFrame(rows)
print("Q1/Q2 per fold (h=24 and h=72 rows; AUCs on the h=72 row):")
print(res.round(3).to_string(index=False))
print("\nmeans:\n", res.groupby("h").mean(numeric_only=True).drop(columns="fold").round(3).to_string())


# Q3: exposure w = clip(c / sigma_hat, 0, 1), c = median sigma_hat on the fold's train part
def seg_sharpe(expo, seg_idx):
    ret = np.exp(r.iloc[seg_idx].values) - 1
    e = np.r_[expo[0], expo[:-1]]                                    # position set at the previous close
    pnl = e * ret - FEE * np.abs(np.diff(np.r_[0.0, e]))
    return summarize(10000 * np.cumprod(1 + pnl))["sharpe"], e.mean()


out = []
for w in wins:
    f = w["fold"]
    tr = np.arange(w["train_start"], w["train_end"])
    naive = np.exp(rv(168).values)                                   # what the traders already see (vol_168h)
    model = np.exp(sigma[f]) if f in sigma else None
    c_naive = np.nanmedian(naive[tr])
    for s in range(w["test_start"], w["test_end"] - SEG + 1, SEG):
        idx = np.arange(s, s + SEG)
        move = np.log(df.close.iloc[idx[-1]] / df.close.iloc[idx[0]])
        row = {"fold": f, "regime": "bull" if move > 0.05 else ("bear" if move < -0.05 else "side")}
        row["bh"], _ = seg_sharpe(np.ones(SEG), idx)
        e = np.clip(c_naive / naive[idx], 0, 1)
        row["vt_naive"], row["expo_naive"] = seg_sharpe(np.nan_to_num(e, nan=1.0), idx)
        sm = model[idx]
        c_model = np.nanmedian(np.exp(fwd(24).values[tr]))           # same scale as the model's target
        e = np.clip(c_model / sm, 0, 1)
        row["vt_model"], row["expo_model"] = seg_sharpe(np.nan_to_num(e, nan=1.0), idx)
        out.append(row)
q3 = pd.DataFrame(out)
print(f"\nQ3: mean segment Sharpe over {len(q3)} dev segments (agent fee {FEE}, no funding)")
print(q3[["bh", "vt_naive", "vt_model", "expo_naive", "expo_model"]].mean().round(3).to_string())
print("\nby regime:\n", q3.groupby("regime")[["bh", "vt_naive", "vt_model", "expo_naive", "expo_model"]].mean().round(3).to_string())
print("\nby fold:\n", q3.groupby("fold")[["bh", "vt_naive", "vt_model"]].mean().round(3).to_string())


# Q3b: relative vol targeting with the HAR forecast (the best Q1 model):
# w = clip(k * median(past 2160 bars of forecast) / forecast, 0, 1), k sets the mean exposure.
print("\nQ3b: HAR relative vol targeting, mean segment Sharpe over 48 dev segments")
har_hat = np.full(len(df), np.nan)
for w in wins:
    tr, _ = split(w, 24)
    te = np.arange(w["test_start"] - 2160, w["test_end"])            # warm-up for the rolling median
    har_hat[te] = np.exp(fitpred(Ridge(1.0), ["rv24", "rv168", "rv720"], fwd(24), tr, te))
ref = pd.Series(har_hat).rolling(2160, min_periods=500).median().values
tab = []
for k in (0.6, 0.8, 1.0, 1.2):
    rs = []
    for w in wins:
        for s in range(w["test_start"], w["test_end"] - SEG + 1, SEG):
            idx = np.arange(s, s + SEG)
            move = np.log(df.close.iloc[idx[-1]] / df.close.iloc[idx[0]])
            sh, ex = seg_sharpe(np.clip(k * ref[idx] / har_hat[idx], 0, 1), idx)
            rs.append({"regime": "bull" if move > 0.05 else ("bear" if move < -0.05 else "side"), "sharpe": sh, "expo": ex})
    t = pd.DataFrame(rs)
    g = t.groupby("regime").sharpe.mean()
    tab.append({"k": k, "sharpe": t.sharpe.mean(), "expo": t.expo.mean(), "bear": g["bear"], "bull": g["bull"], "side": g["side"]})
print(pd.DataFrame(tab).round(3).to_string(index=False))
