"""Live paper trading for the final BTC configuration. No orders, no API key: public Binance data only.

    python paper_trade.py train   # once: train the three arms on the 40,000 bars before LIVE_START
    python paper_trade.py step    # refresh data, replay LIVE_START -> last closed bar, write runs/paper/ledger.csv
    python paper_trade.py loop    # step + plot shortly after every hour
    python paper_trade.py plot    # runs/paper/equity.png, summary.png, summary.csv from the ledger
    python paper_trade.py check   # parity of the live (REST) positioning data against Binance's daily archive

The ledger is a pure function of the data: every step replays the whole live period with the frozen
models, so a restart or a missed hour changes nothing. Fills, fees, slippage and funding are TradingEnv's,
the same as the backtest (the agent trades at the close of the bar it has just seen).
"""
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import coevo_eval as ce
import coevo_node
import coin_data as cd
from build_features import FEATURE_COLUMNS
from download_coin import BASE, fetch

LIVE_START = pd.Timestamp("2026-10-01")       # first live bar; nothing from here on was ever trained or tuned on
TRAIN_SPAN = 40000                            # same as the walk-forward folds
SEEDS = [42, 43, 44, 45, 46, 47]
OUT, LIVE = "../runs/paper", "../data/live"
CFG = coevo_node.CONFIG
SPOT, FAPI = "https://api.binance.com/api/v3", "https://fapi.binance.com"
M5 = 300_000
REST_METRICS = {"sum_open_interest": ("openInterestHist", "sumOpenInterest"),
                "sum_toptrader_long_short_ratio": ("topLongShortPositionRatio", "longShortRatio"),
                "count_long_short_ratio": ("globalLongShortAccountRatio", "longShortRatio"),
                "sum_taker_long_short_vol_ratio": ("takerlongshortRatio", "buySellRatio")}


def get(url):
    for attempt in range(5):
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def ms(t):
    return int(pd.Timestamp(t).timestamp() * 1000)


def refresh():
    """Bring ../data/live up to the last closed hour."""
    os.makedirs(f"{LIVE}/raw", exist_ok=True); os.makedirs(f"{LIVE}/metrics", exist_ok=True)
    now = int(time.time() * 1000)
    # spot klines, in the raw CSV layout coin_data._raw reads
    path = f"{LIVE}/raw/klines.csv"
    rows = pd.read_csv(path, header=None).values.tolist() if os.path.exists(path) else []
    start = int(rows[-1][0]) + 1 if rows else ms(LIVE_START - pd.Timedelta(days=2))
    while True:
        r = get(f"{SPOT}/klines?symbol=BTCUSDT&interval=1h&startTime={start}&limit=1000")
        rows += [x for x in r if int(x[6]) < now]          # closed bars only
        if len(r) < 1000:
            break
        start = int(r[-1][0]) + 1
    pd.DataFrame(rows).to_csv(path, header=False, index=False)
    # funding
    path = f"{LIVE}/funding.csv"
    f = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["fundingTime", "fundingRate"])
    start = int(f.fundingTime.max()) + 1 if len(f) else ms(LIVE_START - pd.Timedelta(days=5))
    r = get(f"{FAPI}/fapi/v1/fundingRate?symbol=BTCUSDT&startTime={start}&limit=1000")
    f = pd.concat([f, pd.DataFrame(r, columns=["fundingTime", "fundingRate"])]) if r else f
    f.to_csv(path, index=False)
    # positioning: Binance's daily archive (about a day behind), then REST for the days it does not cover yet
    for d in pd.date_range(LIVE_START, pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)):
        name = f"BTCUSDT-metrics-{d:%Y-%m-%d}"
        fetch(f"{BASE}/futures/um/daily/metrics/BTCUSDT/{name}.zip", f"{LIVE}/metrics/{name}.csv")
    path = f"{LIVE}/metrics_rest.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["timestamp"])
    done = sorted(glob.glob(f"{LIVE}/metrics/*.csv"))
    arch_end = ms(pd.Timestamp(done[-1][-14:-4]) + pd.Timedelta(days=1)) if done else ms(LIVE_START)
    start = max(arch_end - 12 * M5, now - 29 * 86_400_000, int(old.timestamp.max()) + 1 if len(old) else 0)
    new = None
    for col, (ep, key) in REST_METRICS.items():
        got, s = [], start
        while s < now:
            got += get(f"{FAPI}/futures/data/{ep}?symbol=BTCUSDT&period=5m&limit=500&startTime={s}&endTime={s + 499 * M5}")
            s += 500 * M5
        g = pd.DataFrame({"timestamp": [int(x["timestamp"]) for x in got], col: [float(x[key]) for x in got]})
        new = g if new is None else new.merge(g.drop_duplicates("timestamp"), on="timestamp", how="outer")
    pd.concat([old, new]).drop_duplicates("timestamp", keep="last").sort_values("timestamp").to_csv(path, index=False)


def rest_metrics():
    m = pd.read_csv(f"{LIVE}/metrics_rest.csv")
    m["create_time"] = pd.to_datetime(m.timestamp, unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
    return m.drop(columns="timestamp")


def build_df():
    raw = pd.concat([cd._raw("../data/raw"), cd._raw(f"{LIVE}/raw")])
    raw = raw.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
    files = sorted(glob.glob("../data/external/metrics/csv/*.csv")) + sorted(glob.glob(f"{LIVE}/metrics/*.csv"))
    m = pd.concat([pd.read_csv(f) for f in files] + [rest_metrics()])   # build_coin keeps the first row per time: archive wins
    f = pd.read_csv(f"{LIVE}/funding.csv")
    live = pd.Series(f.fundingRate.astype(float).values,
                     index=pd.DatetimeIndex(pd.to_datetime(f.fundingTime, unit="ms")).astype("datetime64[ns]").round("h"))
    rate = pd.concat([cd._funding("BTCUSDT", None), live]).groupby(level=0).last()
    return cd.build_coin(raw, m, rate)


def prepare(df):
    """Same steps as coevo_eval.prepare for one window: train = the TRAIN_SPAN bars before LIVE_START, test = live."""
    n0 = int(np.searchsorted(df.open_time.values, LIVE_START.to_datetime64()))
    w = {"fold": 0, "train_start": n0 - TRAIN_SPAN, "train_end": n0, "test_start": n0, "test_end": len(df)}
    d = df.copy()
    for h in CFG["rhat"]:
        d[f"rhat{h}"] = ce.rhat_feature(df, list(FEATURE_COLUMNS) + cd.FLOW + cd.POS, [w], h)[0]
    feats = CFG["features"] + [f"rhat{h}" for h in CFG["rhat"]]
    train, test = d.iloc[n0 - TRAIN_SPAN:n0].reset_index(drop=True), d.iloc[n0:].reset_index(drop=True)
    mu, sd = train[feats].mean(), train[feats].std().replace(0, 1)
    for x in (train, test):
        x.loc[:, feats] = ((x[feats] - mu) / sd).clip(-10, 10).fillna(0.0)
    return train, test, feats


def model_dir(arm, seed):
    return f"{OUT}/models/{arm}_seed{seed}"


def rollout(job):
    arm, seed = job
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from trading_env import TradingEnv
    test = ce.DATA[0][1]
    d = f"{model_dir(arm, seed)}/final_traders"
    model = ce.SoftVote([PPO.load(os.path.join(d, f), device="cpu") for f in sorted(os.listdir(d)) if f.endswith(".zip")])
    e = CFG["engine"]
    env = TradingEnv(df=test, feature_columns=ce.FEATS, segments=[(0, len(test))], initial_balance=10000.0,
                     trading_fee=ce.AGENT_FEE, slippage=ce.AGENT_SLIP, trade_penalty=0.0, funding_col="funding_paid",
                     decision_every=e["decision_every"], start_long=e["start_long"], target_position=e["target_position"])
    obs, _ = env.reset(seed=0, options={"segment": 0})
    v, pos, done = [10000.0], [], False
    while not done:
        a = model.predict(obs)[0]
        obs, _, done, _, info = env.step(a)
        v.extend(info["bar_values"]); pos.append(int(info["btc_held"] > 0))
    return job, np.asarray(v) / 10000.0, pos


def step():
    from baselines import buy_and_hold, ma_crossover
    from metrics import summarize
    refresh()
    train, test, feats = prepare(build_df())
    ce.DATA, ce.FEATS, ce.CFG = {0: (train, test)}, feats, CFG
    with mp.get_context("fork").Pool(6) as p:
        res = p.map(rollout, [(a, s) for a in ce.ARMS for s in SEEDS])
    eq = {a: np.mean([v for (arm, _), v, _ in res if arm == a], axis=0) for a in ce.ARMS}
    eq["buy_hold"] = buy_and_hold(test, 10000.0, ce.SPOT_FEE, ce.SPOT_SLIP) / 10000.0
    eq["ma_crossover"] = np.asarray(ma_crossover(test, 10000.0, ce.AGENT_FEE, slippage=ce.AGENT_SLIP)) / 10000.0
    n = min(len(v) for v in eq.values())
    led = pd.DataFrame({"time_utc": (test.open_time + pd.Timedelta(hours=1)).values[:n], "close": test.close.values[:n],
                        **{k: v[:n] for k, v in eq.items()}})
    os.makedirs(OUT, exist_ok=True)
    led.to_csv(f"{OUT}/ledger.csv", index=False)
    longs = {a: np.mean([pos[-1] for (arm, _), _, pos in res if arm == a]) for a in ce.ARMS}
    print(f"paper ledger to {led.time_utc.iloc[-1]} UTC ({n} hourly bars since {LIVE_START:%Y-%m-%d}), BTC {led.close.iloc[-1]:.0f}")
    for k in eq:
        s = summarize(led[k].values)
        print(f"  {k:13s} return {s['total_return_pct']:+6.2f}%  Sharpe {s['sharpe']:+5.2f}  max drawdown {s['max_drawdown_pct']:6.2f}%"
              + (f"  now long in {longs[k]:.0%} of seeds" if k in longs else ""))
    return led


def train():
    refresh()
    tr, test, feats = prepare(build_df())
    ce.DATA, ce.FEATS, ce.CFG = {0: (tr, test)}, feats, CFG
    os.makedirs(f"{OUT}/models", exist_ok=True)
    json.dump({"config": CFG, "features": feats, "live_start": str(LIVE_START), "train_bars": TRAIN_SPAN,
               "train_from": str(tr.open_time.iloc[0]), "train_to": str(tr.open_time.iloc[-1])},
              open(f"{OUT}/contract.json", "w"), indent=1)
    jobs = [(0, a, s, model_dir(a, s)) for a in ce.ARMS[::-1] for s in SEEDS]
    with mp.get_context("fork").Pool(int(os.environ.get("W", 11))) as p:
        for (_, a, s, _), mins in p.imap_unordered(ce.train_job, jobs):
            print(f"trained {a} seed {s} in {mins / 60:.1f} min", flush=True)


NAMES = {"coevolution": "Coevolution", "population": "PBT", "single": "Single-agent PPO",
         "buy_hold": "Buy & hold", "ma_crossover": "MA crossover"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # dataviz default categorical, slots 1-5 (validated)


def plot():
    """runs/paper/equity.png (equity curves + BTC price) and summary.png/.csv (return, Sharpe, max drawdown)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from metrics import summarize
    led = pd.read_csv(f"{OUT}/ledger.csv", parse_dates=["time_utc"])
    ink, ink2, muted, grid, axis, bg = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": ink2, "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": muted, "axes.edgecolor": axis})
    span = f"{led.time_utc.iloc[0]:%d %b} – {led.time_utc.iloc[-1]:%d %b %Y}"
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13.33, 7.5), dpi=150, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12}, facecolor=bg)
    for a in (ax, bx):
        a.set_facecolor(bg); a.grid(axis="y", color=grid, lw=0.8); a.tick_params(length=0)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
    ax.axhline(10000, color=muted, lw=1, ls=(0, (4, 3)))
    rows = []
    for (k, name), c in zip(NAMES.items(), COLORS):
        v = 10000 * led[k].values
        s = summarize(led[k].values)
        rows.append({"method": name, "final_usd": v[-1], "return_pct": s["total_return_pct"],
                     "sharpe": s["sharpe"], "max_drawdown_pct": s["max_drawdown_pct"]})
        ax.plot(led.time_utc, v, color=c, lw=2.6 if k == "coevolution" else 2, solid_capstyle="round",
                label=f"{name}   ${v[-1]:,.0f}  ({s['total_return_pct']:+.1f}%)", zorder=3 if k == "coevolution" else 2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="lower left", frameon=False, fontsize=12, labelcolor=ink2, handlelength=1.6)
    ax.set_title(f"Live paper trading on BTC: $10,000 start, {span}", loc="left", color=ink, fontsize=17, pad=26)
    ax.text(0, 1.02, "Simulated trades on live Binance prices with fees, slippage and funding. "
            "Models frozen on 30 Sep 2026; no real orders.", transform=ax.transAxes, color=ink2, fontsize=11.5)
    bx.plot(led.time_utc, led.close, color=ink2, lw=1.6)
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x / 1000:.0f}k"))
    bx.set_ylabel("BTC price", color=ink2)
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.savefig(f"{OUT}/equity.png", bbox_inches="tight", facecolor=bg)
    t = pd.DataFrame(rows)
    t.round(2).to_csv(f"{OUT}/summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 2.9), dpi=150, facecolor=bg)
    ax.axis("off")
    cells = [[r.method, f"${r.final_usd:,.0f}", f"{r.return_pct:+.1f}%", f"{r.sharpe:+.2f}", f"{r.max_drawdown_pct:.1f}%"]
             for r in t.itertuples()]
    tb = ax.table(cellText=cells, colLabels=["Method", "Value now", "Return", "Sharpe", "Max drawdown"],
                  loc="center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(13); tb.scale(1, 1.9); tb.auto_set_column_width(range(5))
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(grid); cell.set_facecolor(bg)
        cell.get_text().set_color(ink if r == 0 or c == 0 else ink2)
        if r == 0:
            cell.get_text().set_fontweight("bold")
        if r == 1:
            cell.get_text().set_fontweight("bold"); cell.get_text().set_color(ink)
    ax.set_title(f"BTC paper trading, {span} ($10,000 start)", loc="left", color=ink, fontsize=14)
    fig.savefig(f"{OUT}/summary.png", bbox_inches="tight", facecolor=bg)
    print(t.round(2).to_string(index=False))


def check():
    """REST positioning rows must match the archive on the days both cover."""
    refresh()
    a = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{LIVE}/metrics/*.csv"))[-2:]])
    j = a.merge(rest_metrics(), on="create_time", suffixes=("", "_rest"))
    assert len(j) > 6, f"only {len(j)} overlapping rows"
    for c in REST_METRICS:
        err = (j[c] / j[c + "_rest"] - 1).abs()
        print(f"{c:32s} rows {err.notna().sum():3d}  median rel. error {err.median():.2e}  max {err.max():.2e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "step"
    if cmd == "loop":
        while True:
            try:
                step()
                plot()
            except Exception as e:      # a network blip must not kill the loop; the next hour replays everything
                print("step failed:", repr(e))
            sys.stdout.flush()
            time.sleep(3600 - time.time() % 3600 + 90)
    else:
        {"train": train, "step": step, "check": check, "plot": plot}[cmd]()
