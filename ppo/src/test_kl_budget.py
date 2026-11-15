"""Check: adv_kl_budget off = unchanged softmax; on = KL(p || uniform) <= budget."""
import numpy as np
from coevo import env_population as ep

rng = np.random.default_rng(0)
for _ in range(200):
    logits = rng.normal(0, rng.uniform(0.1, 20), size=rng.integers(5, 400))
    ref = np.exp(logits - logits.max()); ref /= ref.sum()
    assert np.array_equal(ep.capped_probs(logits, 0.0), ref)
    for b in (0.05, 0.5, 2.0):
        p = ep.capped_probs(logits, b)
        kl = ep._kl_uniform(p)
        assert kl <= b + 1e-9 and abs(p.sum() - 1) < 1e-9
        if ep._kl_uniform(ref) > b:
            assert kl > b - 1e-3, (kl, b)      # tight: the smallest multiplier, not over-flattened
        else:
            assert np.array_equal(p, ref)
print("kl budget ok")
