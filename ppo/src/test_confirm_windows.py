"""Check the V9 E0-E2 windows WITHOUT touching any test bar: indices and segment count only."""
from coevo_eval import confirm_windows, SEG
w = confirm_windows()
assert [(x["test_start"], x["test_end"]) for x in w] == [(22000, 28000), (28000, 34000), (34000, 40000)]
assert all(x["train_start"] == 0 and x["train_end"] == x["test_start"] for x in w)
n = sum(len(range(0, x["test_end"] - x["test_start"] - SEG + 1, SEG)) for x in w)
assert n == 24, n
print("confirm windows ok: 3 folds, 24 segments")
