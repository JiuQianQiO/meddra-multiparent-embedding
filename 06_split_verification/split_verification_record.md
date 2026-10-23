# Split-Consistency Verification Record

> Purpose: to support the statement in Section 3.5 of the manuscript and in the note to Table 5
> (Section 4.3) that "the splits of the baselines and of the main experiment have been checked
> member by member"; for reviewers and for internal reference.
> **Evidence status: complete ✅** (re-run on the server on 2026-10-05; raw output archived in
> Section 1.2 below).

## 1. Per-member cross-check on the server

### 1.1 First check (2026-10-04)
Per-member cross-checks were run on the server (`<JBI_ROOT>`) for the three baseline groups
B1/B2/B3; all passed and agreed with the main experiment.
Original source: the "split consistency" column of the external-baseline sheet in
`experiment_summary_JBI.xlsx` (in the builder script `build_jbi_xlsx.py`:
`BASE_SPLIT_OK = '✔ verified member-wise against the main experiment (2026-10-04, split_indices.json)'`).

### 1.2 Archived re-run (2026-10-05; this section is the raw output archive ✅)

Environment: server `<HOST>-kkcgq`, `<JBI_ROOT>/BaseLine`.
Command: `python verify_split_alignment.py` (script in this directory).

```
split index : <RUN_ROOT>/split_indices.json
baseline dir: <JBI_ROOT>/BaseLine
CSV         : <DATA_DIR>/MEDDRA_DATA_29_0.csv
cache       : <CACHE_DIR>
reference   : train=97641 val=10837 test=12041
--------------------------------------------------------------------------
[split] per-member cross-check passed ✓  <RUN_ROOT>/split_indices.json
[agree ] train_b1_text_embed.py   train=97641 val=10837 test=12041 total=120519  (2.5s)
[split] per-member cross-check passed ✓  <RUN_ROOT>/split_indices.json
[agree ] train_b2_kge.py          train=97641 val=10837 test=12041 total=120519  (1.5s)
[split] per-member cross-check passed ✓  <RUN_ROOT>/split_indices.json
[agree ] train_b3_graph_ssl.py    train=97641 val=10837 test=12041 total=120519  (1.6s)
--------------------------------------------------------------------------
  train_b1_text_embed.py     (97641, 10837, 12041)  ✔ consistent with the main experiment
  train_b2_kge.py            (97641, 10837, 12041)  ✔ consistent with the main experiment
  train_b3_graph_ssl.py      (97641, 10837, 12041)  ✔ consistent with the main experiment

Conclusion: all consistent -- the baselines need not be re-run; the existing numbers can be used
in the paper directly.
```

Key points:

- The built-in per-member cross-check runs and prints "passed ✓" **inside each baseline, right
  after import and when the split is taken** (the three `[split] per-member cross-check passed ✓`
  lines), after which the sizes agree with the reference values of the main experiment
  (97,641 / 10,837 / 12,041);
- the check is **member-wise**, not merely on sizes, and is carried out on the real data with the
  real graph-construction pipeline;
- side effect worth noting: when B3 was re-run on 2026-10-05 (DGI/GraphMAE, adding validation
  metrics), the split independently produced by `get_split` also agreed with the main experiment —
  so the re-run values (0.8624 / 0.8144 / 0.8619) can be used in the paper.

## 2. Equivalence of the split implementations (synthetic data, 2026-10-05, ✅ passed)

Environment: the same server, `<JBI_ROOT>/SAGE`. Command: `python test_split_equiv.py`
(script in this directory).

```
==== 1. Is the main trainer's split member-wise identical to the baseline get_split? ====
  trial-0        identical=True  disjoint=True  union=all=True  sizes tr/va/te=30347/3382/3756
  trial-1        identical=True  disjoint=True  union=all=True  sizes tr/va/te=58874/6550/7276
  trial-2        identical=True  disjoint=True  union=all=True  sizes tr/va/te=45109/5017/5573
  trial-3        identical=True  disjoint=True  union=all=True  sizes tr/va/te=22620/2522/2800
  trial-4        identical=True  disjoint=True  union=all=True  sizes tr/va/te=29587/3297/3661

==== 2. Does dict ordering cause divergence (string vs integer keys, insertion order)? ====
  Same pool but different dict insertion order -> identical result: False

==== 3. Does test_ratio=0 reproduce the old two-way split? ====
  With test_ratio=0; is test empty: True
  Is train identical to the old two-way split: True
  Is val   identical to the old two-way split: True
```

- **Conclusion**: given the same input, the main training script and the baseline `get_split`
  produce member-wise identical splits; the three sets are disjoint and their union is the full
  set; with `test_ratio=0` the implementation degrades exactly to the old two-way protocol
  (showing that historical behaviour was not changed by accident).
- **Key finding (trial 2)**: the split depends on the insertion order of the SOC grouping dict —
  a different order yields a different split. This is precisely why a **member-wise** check is
  required instead of comparing sizes only; the real-data check in Section 1 covers this point.

## 3. Conclusion

**The statement that the baselines and the main experiment share the same split is fully
supported:**

| Evidence | Content | Status |
|---|---|---|
| ① Per-member cross-check (real data) | B1/B2/B3 all agree with the main experiment (first check 2026-10-04 + archived re-run 2026-10-05) | ✅ raw output archived |
| ② Equivalence of split implementations (synthetic data) | The two implementations agree member-wise, are disjoint, and union to the full set; degradation behaves correctly | ✅ reproducible locally and on the server |
| ③ Corroboration from the B3 re-run | The split independently generated by the re-run has the same sizes as the main experiment | ✅ |

The statements in Section 3.5 and in the note to Table 5 (Section 4.3) can rely directly on this
record; if a reviewer asks for details, this file is provided together with
`verify_split_alignment.py` and `test_split_equiv.py`.
