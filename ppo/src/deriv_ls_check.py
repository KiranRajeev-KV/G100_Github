"""Non-overlapping recheck of the crowding deciles from deriv_probe.py (dev folds only)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from coin_data import fold_windows, load_coin
df = load_coin("BTCUSDT"); wins = fold_windows(df)
fwd = np.log(df.close.shift(-72) / df.close)
rng = np.random.default_rng(0)
for c in ["ls_all", "ls_top", "oi_z30"]:
    if c == "oi_z30":
        s = df.oi_d1.rolling(168).sum(); x = (s - s.rolling(720, min_periods=180).mean()) / s.rolling(720, min_periods=180).std()
    else:
        x = df[c]
    out = []
    for w in wins:
        idx = np.arange(w["test_start"], w["test_end"] - 72, 72)          # non-overlapping 72-bar steps
        xs, ys = x.iloc[idx].values, fwd.iloc[idx].values
        ok = np.isfinite(xs) & np.isfinite(ys); xs, ys = xs[ok], ys[ok]
        tr = x.iloc[w["train_start"]:w["train_end"]]
        lo, hi = np.nanquantile(tr, 0.2), np.nanquantile(tr, 0.8) if tr.notna().sum() > 500 else (np.quantile(xs, 0.2), np.quantile(xs, 0.8))
        rho = pd.Series(xs).corr(pd.Series(ys), method="spearman")
        out.append({"fold": w["fold"], "n": len(xs), "spearman": rho,
                    "q5_minus_q1_%": 100 * (ys[xs >= np.quantile(xs, 0.8)].mean() - ys[xs <= np.quantile(xs, 0.2)].mean())})
    t = pd.DataFrame(out)
    # pooled within-fold rank correlation, bootstrap over non-overlapping obs
    print(c); print(t.round(3).to_string(index=False)); print("mean spearman", round(t.spearman.mean(), 3), "| folds negative", (t.spearman < 0).sum(), "/ 6 | ~se per fold", round(1 / np.sqrt(t.n.mean()), 3), "\n")
