#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline split alignment check: confirm that B1 / B2 / B3 share the same three-way split as the main experiment.

Run on the server:
    python verify_split_alignment.py
Exit code: 0 = all three baselines are member-wise identical to the main experiment; 1 = a mismatch exists (the baseline must be re-run).
"""
import os, sys, importlib.util, time

SPLIT_INDEX = "<RUN_ROOT>/split_indices.json"
BASE_DIR = "<JBI_ROOT>"
MODULES = ["train_b1_text_embed.py", "train_b2_kge.py", "train_b3_graph_ssl.py"]

# Must be set before importing the baseline modules: BaselineConfig reads the env into defaults at class-definition time
os.environ["MEDDRA_SPLIT_INDEX_FILE"] = SPLIT_INDEX
os.environ.setdefault("MEDDRA_CSV_PATH", "<DATA_DIR>/MEDDRA_DATA_29_0.csv")
os.environ.setdefault("MEDDRA_CACHE_DIR", "<CACHE_DIR>")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

REF = (97641, 10837, 12041)          # train / val / test sizes of the main experiment (29.0) as reported in the logs

if not os.path.exists(SPLIT_INDEX):
    sys.exit("Split index not found: %s" % SPLIT_INDEX)

print("split index :", SPLIT_INDEX)
print("baseline dir:", BASE_DIR)
print("CSV      :", os.environ["MEDDRA_CSV_PATH"])
print("cache       :", os.environ["MEDDRA_CACHE_DIR"])
print("reference   : train=%d val=%d test=%d" % REF)
print("-" * 74)

results = {}
for i, fn in enumerate(MODULES):
    path = os.path.join(BASE_DIR, fn)
    if not os.path.exists(path):
        print("[skip] %s not found" % path)
        continue
    spec = importlib.util.spec_from_file_location("bl%d" % i, path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        print("[import failed] %s: %s" % (fn, e))
        results[fn] = "import-error"
        continue
    try:
        cfg = m.BaselineConfig()
        t0 = time.time()
        proc, graph = m.build_processor(cfg)     # build the graph from cache; no training, no encoding
        sp = m.get_split(proc, graph)            # performs the built-in member-wise cross-check
    except AssertionError as e:
        print("[check failed] %s: %s" % (fn, e))
        results[fn] = "mismatch"
        continue
    except Exception as e:
        print("[runtime error] %s: %s" % (fn, e))
        results[fn] = "error"
        continue
    tr, va, te = sp["train"], sp["val"], sp["test"]
    got = (len(tr), len(va), len(te))
    print("[%s] %-24s train=%d val=%d test=%d total=%d  (%.1fs)"
          % ("agree " if got == REF else "DIFFER", fn, got[0], got[1], got[2], sum(got),
             time.time() - t0))
    results[fn] = got

print("-" * 74)
ok = bool(results) and all(v == REF for v in results.values() if isinstance(v, tuple))
for fn, v in results.items():
    if isinstance(v, tuple):
        print("  %-26s %s  %s" % (fn, v, "OK  consistent with the main experiment" if v == REF else "XX  INCONSISTENT with the main experiment"))
    else:
        print("  %-26s %s" % (fn, v))
print("\nConclusion: %s" % ("all consistent -- the baselines need not be re-run; the existing values can be used in the paper."
                     if ok else "a mismatch exists -- re-run the corresponding baseline with MEDDRA_SPLIT_INDEX_FILE."))
sys.exit(0 if ok else 1)
