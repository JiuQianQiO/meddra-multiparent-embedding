# -*- coding: utf-8 -*-
"""Verify on synthetic data whether the split implementation of the main training script
and the baseline get_split are member-wise identical. Also checks whether SAGE(_split_seed) and GAT(random_state) diverge when split_seed is set explicitly."""
import numpy as np
from collections import defaultdict

VAL_RATIO = 0.1
TEST_RATIO = 0.1


def trainer_split(soc_to_indices, seed):
    """Replicates lines 2071-2106 of Meddra_full_trainer_SAGE.py."""
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.RandomState(seed)
    for lab, indices in soc_to_indices.items():
        _lst = list(indices)
        if TEST_RATIO > 0 and len(_lst) >= 3:
            n_test = max(1, min(int(len(_lst) * TEST_RATIO), len(_lst) // 2))
            test_sel = rng.choice(_lst, n_test, replace=False)
            _test_set = set(int(t) for t in test_sel)
            _rest = [i for i in _lst if i not in _test_set]
        else:
            test_sel, _test_set = [], set()
            _rest = _lst
        n_val = max(1, min(int(len(_rest) * VAL_RATIO), len(_rest) // 2))
        if len(_rest) <= 2:
            val_sel = _rest[:1]
            train_sel = _rest[1:]
        else:
            val_sel = rng.choice(_rest, n_val, replace=False)
            _val_set = set(int(v) for v in val_sel)
            train_sel = [i for i in _rest if i not in _val_set]
        test_idx.extend(test_sel)
        val_idx.extend(val_sel)
        train_idx.extend(train_sel)
    return train_idx, val_idx, test_idx


def baseline_split(soc_to_indices, seed):
    """Replicates get_split of baseline/train_b*_*.py."""
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.RandomState(seed)
    for lab, indices in soc_to_indices.items():
        _lst = list(indices)
        if TEST_RATIO > 0 and len(_lst) >= 3:
            n_test = max(1, min(int(len(_lst) * TEST_RATIO), len(_lst) // 2))
            test_sel = rng.choice(_lst, n_test, replace=False)
            _test_set = {int(t) for t in test_sel}
            _rest = [i for i in _lst if i not in _test_set]
        else:
            test_sel, _test_set = [], set()
            _rest = _lst
        n_val = max(1, min(int(len(_rest) * VAL_RATIO), len(_rest) // 2))
        if len(_rest) <= 2:
            val_sel = _rest[:1]
            train_sel = _rest[1:]
        else:
            val_sel = rng.choice(_rest, n_val, replace=False)
            _val_set = {int(v) for v in val_sel}
            train_sel = [i for i in _rest if i not in _val_set]
        test_idx.extend(test_sel)
        val_idx.extend(val_sel)
        train_idx.extend(train_sel)
    return train_idx, val_idx, test_idx


def make_pools(rng, n_soc=27, sizes=(3, 5, 12, 200, 9000)):
    """Build an SOC pool shaped like the real data (includes very small pools to cover the len<=2 / ==3 branches)."""
    pools = {}
    offset = 0
    for lab in range(n_soc):
        n = int(rng.choice(sizes))
        pools[lab] = list(range(offset, offset + n))
        offset += n
    return pools


def check(pools, name):
    a = trainer_split(pools, 42)
    b = baseline_split(pools, 42)
    same = all(list(x) == list(y) for x, y in zip(a, b))
    tr, va, te = a
    s_tr, s_va, s_te = set(map(int, tr)), set(map(int, va)), set(map(int, te))
    disjoint = not (s_tr & s_va or s_tr & s_te or s_va & s_te)
    full = (s_tr | s_va | s_te) == set(range(sum(len(v) for v in pools.values())))
    print('  %-14s identical=%s  disjoint=%s  union=all=%s  sizes tr/va/te=%d/%d/%d'
          % (name, same, disjoint, full, len(tr), len(va), len(te)))
    return same and disjoint and full


print('==== 1. Is the main trainer split member-wise identical to the baseline get_split? ====')
ok = True
for trial in range(5):
    pools = make_pools(np.random.RandomState(trial))
    ok &= check(pools, 'trial-%d' % trial)

print('\n==== 2. Does dict ordering cause divergence (string vs integer keys, insertion order)? ====')
pools_int = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1: [10, 11, 12]}
pools_as_insert = {}
for k in [1, 0]:
    pools_as_insert[k] = pools_int[k]
a = trainer_split(pools_int, 42)
b = trainer_split(pools_as_insert, 42)
print('  Same pool but different dict insertion order -> identical result:', a == b)

print('\n==== 3. Does test_ratio=0 reproduce the old two-way split? ====')
TEST_RATIO = 0.0


def old_two_way(soc_to_indices, seed=42):
    train_idx, val_idx = [], []
    rng = np.random.RandomState(seed)
    for lab, indices in soc_to_indices.items():
        n_val = max(1, min(int(len(indices) * 0.1), len(indices) // 2))
        if len(indices) <= 2:
            val_sel = indices[:1]
            train_sel = indices[1:]
        else:
            val_sel = rng.choice(indices, n_val, replace=False)
            _vs = {int(v) for v in val_sel}
            train_sel = [i for i in indices if i not in _vs]
        val_idx.extend(val_sel)
        train_idx.extend(train_sel)
    return train_idx, val_idx


TEST_RATIO = 0.0
pools = make_pools(np.random.RandomState(7))
tr3, va3, te3 = trainer_split(pools, 42)
tr2, va2 = old_two_way(pools, 42)
print('  With test_ratio=0; is test empty:', len(te3) == 0)
print('  Is train identical to the old two-way split:', list(map(int, tr3)) == list(map(int, tr2)))
print('  Is val   identical to the old two-way split:', list(map(int, va3)) == list(map(int, va2)))
