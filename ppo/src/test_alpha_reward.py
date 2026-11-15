"""Check: with alpha_reward, always-long earns ~0 reward after the entry bar and always-flat earns minus the market's log return."""
import numpy as np, pandas as pd
from trading_env import TradingEnv
rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), "f": rng.normal(0, 1, n), "funding_paid": 1e-5})
kw = dict(df=df, feature_columns=["f"], segments=[(0, n)], funding_col="funding_paid", decision_every=4, start_long=True, target_position=True)
def run(action, alpha):
    env = TradingEnv(alpha_reward=alpha, **kw); env.reset(seed=0, options={"segment": 0})
    r, done = [], False
    while not done:
        _, x, done, _, _ = env.step(action); r.append(x)
    return np.array(r)
long_a, flat_a, long_0, flat_0 = run(1, True), run(0, True), run(1, False), run(0, False)
assert np.abs(long_a).max() < 1e-9, np.abs(long_a).max()
assert np.allclose((flat_0 - flat_a)[1:], long_0[1:], atol=1e-9)       # the subtracted term is the always-long return
assert np.allclose(run(0, False), flat_0) and not np.allclose(flat_a, flat_0)
print("alpha_reward ok", round(float(flat_a.sum()), 4), round(float(-long_0.sum()), 4))
