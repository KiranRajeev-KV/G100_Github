"""One-node diagnosis for the V11 loop: gaps by regime / fold, exposure, timing, adversary dynamics.

    python v11_diag.py v11a-regret-gate
"""
import glob, sys
import numpy as np
import pandas as pd
from evaluate import paired_bootstrap
from coevo_v9 import BASE, regime

d = f"{BASE}/{sys.argv[1]}/BTCUSDT"
s = pd.read_csv(f"{d}/per_segment.csv")
s["regime"] = s.move.map(regime)
c, p = s.coevolution, s.population
print("means  " + "  ".join(f"{a} {s[a].mean():+.3f}" for a in ["coevolution", "population", "single", "buy_hold", "ma_crossover"]))
print("c - x  " + "  ".join(f"{o} {(c - s[o]).mean():+.3f} (P {paired_bootstrap(c.values, s[o].values):.3f})"
                            for o in ["population", "single", "buy_hold", "ma_crossover"]))
print(f"2x     coevo {s.coevolution_2x.mean():+.3f}  pop {s.population_2x.mean():+.3f}  B&H {s.buy_hold_2x.mean():+.3f}")
print("regime c - p  " + "  ".join(f"{g} {(c - p)[s.regime == g].mean():+.2f} (n {(s.regime == g).sum()})" for g in ("bear", "bull", "sideways")))
print("expo c / p    " + "  ".join(f"{g} {s.coevolution_expo[s.regime == g].mean():.2f} / {s.population_expo[s.regime == g].mean():.2f}"
                                   for g in ("bear", "bull", "sideways")) + f"  all {s.coevolution_expo.mean():.2f} / {s.population_expo.mean():.2f}")
print("timing        " + "  ".join(f"{a} {s[a + '_expo'][s.move > 0].mean() - s[a + '_expo'][s.move < 0].mean():+.3f}" for a in ("coevolution", "population", "single")))
f = s.groupby("fold")[["coevolution", "population"]].mean()
print("per fold c - p " + "  ".join(f"{k}: {v:+.2f}" for k, v in (f.coevolution - f.population).items())
      + f"   worst fold c {f.coevolution.min():+.2f} / p {f.population.min():+.2f}")
adv = pd.concat([pd.read_csv(x) for x in glob.glob(f"{d}/fold*/coevolution_seed*/adversary_params.csv")])
g = adv.groupby("generation")
print("adversary by generation (entropy, raw fitness of evaluated):")
print("  " + "  ".join(f"{k}: {v:.2f}/{adv[(adv.generation == k) & adv.evaluated].raw_fitness.mean():+.2f}" for k, v in g.selection_entropy.mean().items()))
last = adv[(adv.generation == adv.generation.max()) & (adv["rank"] < 3)]
print("top-3 final weights  " + "  ".join(f"{k[2:]} {last[k].mean():+.2f}" for k in adv.columns if k.startswith("w_")))
print("served regime mix (last 3 gens): " + "  ".join(f"{k[5:]} {adv[adv.generation >= adv.generation.max() - 2][k].mean():.2f}" for k in ("frac_bear", "frac_bull", "frac_sideways")))
