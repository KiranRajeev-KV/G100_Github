"""Check: regret_gate reference >= always-long on every pool window, and the gates differ from always-long."""
import numpy as np
import coevo_eval as ce, coevo_node
from segments import build_segment_pool
from coevo.engine import CoevolutionEngine

cfg = coevo_node.CONFIG
data, feats, wins = ce.prepare(ce.SYM, cfg)
ce.CFG = cfg
train, _ = data[0]
pool = build_segment_pool(train.iloc[:-ce.VAL_SPAN].reset_index(drop=True), segment_length=ce.SEG, stride=ce.POOL_STRIDE)
import tempfile
eng = CoevolutionEngine(ce.engine_config(cfg, "coevolution", 42), pool, feats, train.iloc[-ce.VAL_SPAN:], tempfile.mkdtemp())
ref = eng._ref_by_start
one = eng._gate_reference(cols=())           # always-long only
assert len(ref) == len(pool.segments) and all(ref[k] >= one[k] - 1e-12 for k in ref)
g = np.array([ref[k] - one[k] for k in ref])
print(f"windows {len(g)}  always-long {np.mean(list(one.values())):+.3f}  reference {np.mean(list(ref.values())):+.3f}  gate better on {np.mean(g > 0):.0%}")
assert (g > 0).mean() > 0.1

# adv_gate_desc: 8-column descriptors and genomes when on, 5 when off (engine built again afterwards resets it)
import dataclasses
from coevo import env_population as ep
pool2 = build_segment_pool(train.iloc[:-ce.VAL_SPAN].reset_index(drop=True), segment_length=ce.SEG, stride=ce.POOL_STRIDE)
on = CoevolutionEngine(dataclasses.replace(ce.engine_config(cfg, "coevolution", 42), adv_gate_desc=True, resume=False),
                       pool2, feats, train.iloc[-ce.VAL_SPAN:], tempfile.mkdtemp())
assert pool2.descriptor_matrix.shape[1] == 8 and all(len(e.weights) == 8 for e in on.env_pop)
assert len(on._adv_select(on.env_pop[0], 4)) == 4 and len(ep.evolve(on.env_pop, on.rng, fresh_frac=0.25)[-1].weights) == 8
pool3 = build_segment_pool(train.iloc[:-ce.VAL_SPAN].reset_index(drop=True), segment_length=ce.SEG, stride=ce.POOL_STRIDE)
CoevolutionEngine(dataclasses.replace(ce.engine_config(cfg, "coevolution", 42), adv_gate_desc=False, resume=False), pool3, feats, train.iloc[-ce.VAL_SPAN:], tempfile.mkdtemp())
assert ep.N_DESC == 5 and pool3.descriptor_matrix.shape[1] == 5
print("adv_gate_desc ok")
