# Consolidated Result Tables

> Data source: `test_metrics_*.json`, `result_B*.json`, `REVIEW_SEED_metrics.csv`,
> `RULE_BASELINE_NEIGHBOR.json`, `MULTIAXIAL_SUMMARY.json`, `INDUCTIVE_SEEDS.json` and
> `PAPER_NUMBERS.md` under the analysis-output directory (`analysis_out_20261029/analysis_out/`).
>
> **Protocol**: all main results use the independent test set of the three-way protocol
> (train / val / test). MedDRA 29.0: train 97,641 / val 10,837 / test 12,041
> (`test_ratio=0.1`, `val_ratio=0.1`, `split_seed=42`); MedDRA 29.1: 98,094 / 10,889 / 12,096.
> The split is verified member-by-member (see `06_split_verification/`).

---

## Table 1 · Data statistics and multiaxiality (MedDRA 29.0)

| Level | Nodes | Multi-parent nodes / total | Multi-SOC nodes / total | Test set: multi-SOC / total |
|---|---|---|---|---|
| SOC | 27 | — | — | — |
| HLGT | 337 | 17/337 (5.04%) | 17/337 (5.04%) | 1/35 (2.86%) |
| HLT | 1,739 | 18/1,739 (1.04%) | 119/1,739 (6.84%) | 14/177 (7.91%) |
| PT | 27,361 | 10,350/27,361 (37.83%) | 11,650/27,361 (42.58%) | 1,162/2,738 (42.44%) |
| LLT | 91,082 | 0/91,082 | 42,539/91,082 (46.70%) | 4,226/9,091 (46.49%) |
| **Total** | 120,519 | — | 54,325/120,519 (45.08%) | 5,403/12,041 (44.87%) |

> Multi-parent and multi-SOC overlap heavily: every multi-parent PT is also multi-SOC
> (10,350/10,350), and 88.84% of multi-SOC PTs are multi-parent (1,300 are single-parent multiaxial).

---

## Table 2 · Main and external-baseline results (held-out-label test set, 12,041 nodes)

HLGT 35 / HLT 177 / PT 2,738 / LLT 9,091. Per-level columns are Macro-F1 within the level.

| Category | Model | Acc | Macro-F1 | Weighted-F1 | HLGT F1 | HLT F1 | PT F1 | LLT F1 |
|---|---|---|---|---|---|---|---|---|
| **Proposed** | **SAGE + PubMedBERT + CL (C1V1)** | **0.9932** | **0.9892** | **0.9932** | 0.8784 | 0.8954 | 0.9742 | 0.9946 |
| GNN control | GAT + PubMedBERT + CL | 0.9911 | 0.9869 | 0.9911 | 0.8784 | 0.9030 | 0.9700 | 0.9928 |
| no-graph | NoGNN + PubMedBERT | 0.8345 | 0.7793 | 0.8346 | 0.7077 | 0.7125 | 0.7495 | 0.7828 |
| no-graph | NoGNN + SapBERT | 0.8915 | 0.8533 | 0.8913 | 0.7032 | 0.7545 | 0.8057 | 0.8656 |
| no-graph | NoGNN + CODER | 0.8947 | 0.8571 | 0.8947 | 0.6500 | 0.8433 | 0.8321 | 0.8628 |
| training-free kNN | kNN over BGE | 0.9372 | 0.9168 | 0.9373 | — | — | — | — |
| training-free kNN | kNN over E5 | 0.9244 | 0.8987 | 0.9245 | — | — | — | — |
| text embedding | B1-BGE (probe) | 0.8684 | 0.8312 | 0.8679 | — | — | — | — |
| text embedding | B1-E5 (probe) | 0.8526 | 0.8114 | 0.8521 | — | — | — | — |
| structure only | B2-TransE | 0.2100 | 0.0911 | 0.1538 | — | — | — | — |
| structure only | B2-RotatE | 0.2852 | 0.2101 | 0.2599 | — | — | — | — |
| structure only | B2-ComplEx | 0.1849 | 0.0131 | 0.0618 | — | — | — | — |
| graph SSL | B3-DGI | 0.8624 | 0.8144 | 0.8619 | — | — | — | — |
| graph SSL | B3-GraphMAE | 0.8410 | 0.7827 | 0.8399 | — | — | — | — |

> All rows are single seed-42 runs; multi-seed means for the main models are in Table A (below) and
> Appendix A.4. Key gaps: the proposed method leads NoGNN+PubMedBERT (same encoder) by +15.87/+20.99
> (Acc/Macro-F1) and the strongest text-only reference kNN-BGE by +5.60/+7.24.

---

## Table 3 · Contrastive-loss ablation (C1–C7 + V2; single seed-42)

| Config | Removed | Acc | ΔAcc | Macro-F1 | ΔF1 | PT intra (val) |
|---|---|---|---|---|---|---|
| C1V1 | — (reference) | 0.9932 | 0 | 0.9892 | 0 | 0.8210 |
| C2 | secondary-parent siblings | 0.9940 | +0.08 | 0.9915 | +0.23 | 0.8127 |
| C3 | parent-child | 0.9903 | −0.29 | 0.9846 | −0.46 | 0.8602 |
| C4 | hard negatives | 0.9929 | −0.03 | 0.9898 | +0.06 | 0.8223 |
| C5 | positive-negative exclusion | 0.9933 | +0.01 | 0.9899 | +0.07 | 0.8223 |
| C6 | primary-parent siblings | 0.9944 | +0.12 | 0.9919 | +0.27 | 0.8046 |
| C7 | primary-parent siblings + parent-child | 0.9867 | −0.65 | 0.9798 | −0.94 | 0.8614 |
| V2 | entire CL (features unchanged) | 0.9943 | +0.11 | 0.9912 | +0.20 | 0.8028 |

> Only C7 (−1.38 pp over five seeds) exceeds the 0.93 pp cross-configuration threshold and V1's ±2 sd.
> Removing CL (V2) is neutral (+0.02 pp over five seeds), so CL is not shown to improve classification.

---

## Table 4 · Two-level subgroup attribution (percentage points)

| Model | Parent-chain net effect | 2-D joint | PT parent∈ − ∉ | Parent component | Child component |
|---|---|---|---|---|---|
| Proposed | +1.44 | +2.23 | +0.84 | −0.08 | +2.31 |
| GAT | +1.03 | +1.31 | +0.62 | −0.01 | +1.32 |
| NoGNN + PubMedBERT | −0.78 | +1.11 | −0.18 | −2.71 | +3.81 |
| NoGNN + SapBERT | +0.17 | +1.00 | −0.56 | −3.58 | +4.58 |
| NoGNN + CODER | −0.72 | +0.38 | −1.27 | −3.45 | +3.83 |

> Parent-chain groups are 10,749/1,238 (PT level 2,235/503); the 2-D joint is n=5,642. The remaining
> 54 test nodes are 35 HLGT (parent = SOC, outside the diagnosis) + 19 HLT with no primary-parent record.

---

## Table 5 · Inductive evaluation (whole-HLT-subtree hold-out, disjoint model selection)

| Setting | Model | Acc | Macro-F1 | vs no-graph baseline (pp) |
|---|---|---|---|---|
| A (EDGE-MASK) | IND-SAGE-MASK | 0.4228 | 0.2023 | −32.18 |
| A (EDGE-MASK) | IND-NoGNN-PubMedBERT | 0.7445 | 0.6339 | — (reference) |
| B (EDGE-KEEP) | IND-SAGE-KEEP | **0.9280 ± 0.0013** | **0.8512 ± 0.0052** | **+18.34** |

> IND-SAGE-KEEP is mean ± sample sd over five training seeds; the other two rows are single runs.
> KEEP − ancestor-label rule (I5 = 0.8172/0.7167) = +11.08 pp. Model selection is disjoint (weights
> selected on non-held-out nodes only); first-round values are reported in Appendix B.7.

---

## Rule baselines and kNN (deterministic, same test split)

| Rule | Information used | Acc | Macro-F1 |
|---|---|---|---|
| R1 | node's own label fields | 1.0000 | 1.0000 |
| R2 | topology only (unique ancestor, abstain) | 0.5481 | 0.6485 |
| R3/R4 | topology only (first / majority ancestor) | 0.7827 | 0.7193 |
| R5 | labelled one-hop neighbours | 0.9610 | 0.9421 |
| **R6** | labelled children → parent → sibling-LLT cascade | **0.9917** | **0.9883** |

> R6 covers 97.53% of test nodes (fallback to R3). The proposed method exceeds R6 by +0.15/+0.09 pp
> single-run (+0.23/+0.24 pp over five seeds); the transductive accuracy therefore mostly reflects the
> primary axis recoverable from labelled neighbours.

---

## Subgroup, SOC-control, multi-seed, and twin-LLT summaries

**Multiaxiality (V1, n=5)**: single-axis Acc 0.9981 ± 0.0001, multiaxial Acc 0.9889 ± 0.0014,
multiaxial Macro-F1 0.9813 ± 0.0028. Test set is 44.87% multi-SOC.

**SOC-name / structure controls (overall Macro-F1, n=3, Δ vs V1 mean)**: E1A +0.07, E1B −0.03,
SOCDROP −0.06 — SOC names, identity and structure are not a classification shortcut.

**Multi-seed (Acc / Macro-F1)**: V1 0.9940±0.0006 / 0.9907±0.0011 (n=5); V2 0.9939±0.0009 /
0.9909±0.0010 (n=5); C7 0.9854±0.0019 / 0.9769±0.0029 (n=5); V3 0.9807±0.0056 / 0.9692±0.0094
(n=5); V4 0.9883±0.0027 / 0.9817±0.0042 (n=5); GAT 0.9919±0.0008 / 0.9884±0.0013 (n=3);
NoGNN+PubMedBERT 0.8336±0.0025 / 0.7785±0.0033 (n=3).

**Twin-LLT leakage**: 81.3% of test PTs (2,226/2,738) have an identically coded and named LLT in the
training set. Splitting test PTs by twin-LLT visibility gives 0.9870 vs 0.9766 accuracy (+1.04 pp,
z=1.75, two-sided p=0.08), i.e. at most +0.19 pp of the overall 0.9932. The reverse channel is absent.

---

## Training-cost benchmark (Tesla T4)

| Model | Params | Memory | Peak GPU | Time/epoch |
|---|---|---|---|---|
| SAGE (C1V1) | 4.28M | 3.5G | 3.6G | 288 s |
| SAGE (V2) | 4.28M | 3.4G | 3.0G | 106 s |
| GAT | 7.80M | 3.5G | 11.2G | 287 s |
| NoGNN (three variants) | 0.99M | 3.3–3.4G | 1.3–1.4G | 106–114 s |
| IND-SAGE | 4.28M | 3.5G | 3.6G | 304 s |

---

> Every number above traces to `07_run_artifacts/` (metrics / histories / diagnostics / provenance)
> or the analysis-output directory; see `PAPER_NUMBERS.md` for the place-by-place source mapping.
