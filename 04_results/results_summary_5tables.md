# Consolidated Result Tables (5 tables)

> Data source: `test_metrics_*.json`, `result_B*.json`, `inductive_holdout.json` and the `run_*.log`
> files under each model directory of the run-artefact root.
>
> **Protocol (important)**: all main results are obtained on the **independent test set of the
> three-way protocol (train / val / test)** — under MedDRA 29.0, train 97,641 / val 10,837 /
> test 12,041 (`test_ratio=0.1`, `val_ratio=0.1`, `split_seed=42`); under 29.1, train 98,094 /
> val 10,889 / test 12,096. The `num_test_nodes` field of `test_metrics_*.json` gives the number of
> independent test nodes. The three noGNN variants were re-run under the three-way protocol on
> 2026-10-04 (latest logs `run_noGNN_*_20261004_*.log`).

---

## Table 1 · Main experiment: MedDRA hierarchical classification (three-way independent test set, 12,041 nodes under 29.0)

HLGT 35 / HLT 177 / PT 2,738 / LLT 9,091 (29.0 test set). All GNN models use PubMedBERT
initialisation and full contrastive learning unless annotated otherwise.

| Model | Init | Acc | Macro-F1 | Weighted-F1 | HLGT F1 | HLT F1 | PT F1 | LLT F1 |
|---|---|---|---|---|---|---|---|---|
| **SAGE (proposed / C1V1)** | PubMedBERT | **0.9932** | 0.9892 | 0.9932 | 0.8784 | 0.8954 | 0.9742 | 0.9946 |
| SAGE − primary-parent siblings (C6) | PubMedBERT | 0.9944 | **0.9919** | 0.9944 | 0.8784 | 0.9052 | 0.9832 | 0.9958 |
| SAGE − secondary-parent siblings (C2) | PubMedBERT | 0.9940 | 0.9915 | 0.9940 | 0.8784 | 0.9120 | 0.9795 | 0.9960 |
| SAGE without contrastive learning (V2) | PubMedBERT | 0.9943 | 0.9912 | 0.9943 | 0.8784 | 0.9172 | 0.9800 | 0.9954 |
| SAGE − hard negatives (C4) | PubMedBERT | 0.9929 | 0.9898 | 0.9929 | 0.8784 | 0.9334 | 0.9787 | 0.9936 |
| SAGE − positive/negative exclusion (C5) | PubMedBERT | 0.9933 | 0.9899 | 0.9933 | 0.8784 | 0.9120 | 0.9770 | 0.9945 |
| SAGE (MedDRA 29.1) * | PubMedBERT | 0.9942 | 0.9919 | 0.9942 | 0.9062 | 0.8842 | 0.9799 | 0.9965 |
| SAGE − parent-child term (C3) | PubMedBERT | 0.9903 | 0.9846 | 0.9903 | 0.8784 | 0.9220 | 0.9636 | 0.9912 |
| SAGE random init + CL (V4) | random | 0.9890 | 0.9820 | 0.9889 | 0.8675 | 0.8944 | 0.9549 | 0.9902 |
| SAGE − primary-parent siblings − parent-child (C7) | PubMedBERT | 0.9867 | 0.9798 | 0.9867 | 0.8784 | 0.9024 | 0.9617 | 0.9858 |
| SAGE random init, no CL (V3) | random | 0.9747 | 0.9578 | 0.9746 | 0.8675 | 0.8785 | 0.9290 | 0.9659 |
| **GAT** | PubMedBERT | 0.9911 | 0.9869 | 0.9911 | 0.8784 | 0.9030 | 0.9700 | 0.9928 |
| noGNN-CODER | CODER | 0.8947 | 0.8571 | 0.8947 | 0.6500 | 0.8433 | 0.8321 | 0.8628 |
| noGNN-SapBERT | SapBERT | 0.8915 | 0.8533 | 0.8913 | 0.7032 | 0.7545 | 0.8057 | 0.8656 |
| noGNN-PubMedBERT | PubMedBERT | 0.8345 | 0.7793 | 0.8346 | 0.7077 | 0.7125 | 0.7495 | 0.7828 |

> Note: the HLGT macro-F1 of most SAGE/GAT models is constant at 0.8784 (the HLGT test set has only
> 35 samples, and the models make nearly identical predictions at that level).
> *The MedDRA 29.1 row is a different data version (121,079 nodes, test set 12,096); its node
> enumeration differs from 29.0, so its per-level sample sizes must not be mixed with those of the
> 29.0 rows. That row serves only the cross-version comparison (see Section 4.6 of the manuscript).
> Key gaps: the proposed method (C1V1) leads noGNN-PubMedBERT, which shares the same encoder, by
> **+15.87 / +20.99** points (Acc / Macro-F1), and the strongest no-graph baseline noGNN-CODER by
> +9.85 / +13.21 points.

---

## Table 2 · Ablation: independent contribution of each contrastive-loss term (SAGE)

| Ablation | Removed term | Δ Macro-F1 (vs C1V1 = 0.9892) | Observation |
|---|---|---|---|
| C1V1 | — (full baseline) | 0 (0.9892) | — |
| C6 | primary-parent siblings | +0.0027 | redundant, slight increase |
| C2 | secondary-parent siblings | +0.0023 | redundant, slight increase |
| V2 | entire contrastive learning | +0.0020 | CL gain beyond graph structure ≈ 0 |
| C4 | hard negatives | +0.0006 | almost no effect |
| C5 | positive/negative exclusion | +0.0007 | almost no effect |
| C3 | parent-child term | **−0.0046** | **the only clearly positive contributor** |
| C7 | primary-parent siblings + parent-child | −0.0094 | cumulative effect of removing the parent-child term |

> Note: Δ is obtained by **subtracting the 4-decimal values shown in the table** (consistent with
> the manuscript). Using full-precision raw values gives C6 +0.28, C2 +0.24, V2 +0.21, C4 +0.06,
> C5 +0.07, C3 −0.45, C7 −0.94 points; the two conventions differ only in the last digit and do not
> change the direction or ranking of any conclusion.

**Conclusion**: among the five terms, only the **parent-child term** yields a substantial
classification gain; the sibling terms, hard negatives and exclusion are redundant or even
slightly harmful under the current setting.

---

## Table 3 · Joint ablation of initialisation and contrastive learning (V series)

| Model | Init | CL | Macro-F1 | Δ (vs C1V1) |
|---|---|---|---|---|
| C1V1 | PubMedBERT | ✓ | 0.9892 | baseline |
| V2 | PubMedBERT | ✗ | 0.9912 | +0.0020 |
| V4 | random | ✓ | 0.9820 | −0.0072 |
| V3 | random | ✗ | 0.9578 | −0.0314 |

**Conclusion**: the contribution of pretrained-language-model initialisation (≈2–3 F1) far exceeds
that of contrastive learning (≈0–0.2 F1); random initialisation needs 300+ epochs to converge.

---

## Table 4 · Inductive generalisation: held-out HLT subtrees (174 HLTs with all their descendants, 18,567 nodes)

Held-out composition: HLT 174 / PT 4,157 / LLT 14,236 = 18,567 (training set 101,952; the two sum
to 120,519). HLGT and SOC nodes are not in the held-out set, so the HLGT level has no evaluation
samples (recorded as 0) — this does not mean the level was untrained.

| Model | Held-out mode | Acc | Macro-F1 | HLT Acc | PT Acc | LLT Acc |
|---|---|---|---|---|---|---|
| IND-SAGE-KEEP | edges kept | **0.9333** | **0.8682** | 0.9253 | 0.9310 | 0.9340 |
| IND-GAT-KEEP | edges kept | 0.9315 | 0.8617 | 0.9195 | 0.9298 | 0.9321 |
| IND-noGNN (PubMedBERT) | no edges | 0.7453 | 0.6299 | 0.8276 | 0.7640 | 0.7388 |
| IND-GAT-MASK | edges removed | 0.5867 | 0.3454 | 0.7586 | 0.5850 | 0.5851 |
| IND-SAGE-MASK | edges removed | 0.5487 | 0.3580 | 0.6839 | 0.4934 | 0.5632 |

> Note: the HLT / PT / LLT columns are **per-level accuracies** (unlike the same-named columns of
> Table 1, which are per-level Macro-F1). The run logs output both per-level accuracy and per-level
> Macro-F1 (e.g. IND-SAGE-KEEP at HLT: Acc 0.9253 / Macro-F1 0.8629); this table consistently uses
> the former. The corresponding per-level Macro-F1 values are: IND-SAGE-KEEP 0.8629 / 0.8677 /
> 0.8679, IND-GAT-KEEP 0.8340 / 0.8663 / 0.8594, IND-noGNN 0.7293 / 0.6543 / 0.6201,
> IND-GAT-MASK 0.5558 / 0.3311 / 0.3401, IND-SAGE-MASK 0.4334 / 0.3060 / 0.3530.

**Conclusion**: GNNs with edges kept (0.93) ≫ noGNN without edges (0.75) ≫ GNNs with edges removed
(0.55). Generalisation comes from graph message passing rather than from node text features.

---

## Table 5 · Seed stability (SEED43 re-run, three-way independent test set)

| Model | seed=42 | seed=43 | Δ Macro-F1 |
|---|---|---|---|
| C7 (no primary-parent siblings, no parent-child) | 0.9798 | 0.9732 | −0.0066 |
| V3 (random, no CL) | 0.9578 | 0.9755 | +0.0177 |

> The main models (C1V1/C6/C2, etc.) are highly stable; the two "weakened" configurations show small
> variation when the seed changes (0.7–1.8 F1). V3 varies more because its random features change
> with the seed.

---

## Training-cost benchmark (Tesla T4, standardised)

| Model | Memory | GPU memory | Time (s) |
|---|---|---|---|
| SAGE (C1V1 full) | 3.5G | 3.6G | 288 |
| SAGE (V2/V3/V4) | 2.5–3.4G | 2.6–3.4G | 104–284 |
| GAT | 3.5G | 11.2G | 287 |
| noGNN (three variants) | 3.3–3.4G | 1.3–1.4G | 106–114 |
| IND-SAGE | 3.5G | 3.6G | 304 |
| IND-GAT | 3.5G | 11.3–11.7G | 298–310 |

---

## External baselines (B1–B3, three-way protocol, test set)

| Category | Model | Acc | Macro-F1 | Weighted-F1 | Note |
|---|---|---|---|---|---|
| Structure only | B2-TransE | 0.2100 | 0.0911 | 0.1538 | PyKEEN, frozen embeddings + probe |
| Structure only | B2-RotatE | 0.2852 | 0.2101 | 0.2599 | same as TransE |
| Structure only | B2-ComplEx | 0.1849 | 0.0131 | 0.0618 | same as TransE |
| Strong text | B1-BGE | 0.8684 | 0.8312 | 0.8679 | kNN variant 0.9372 |
| Strong text | B1-E5 | 0.8526 | 0.8114 | 0.8521 | kNN variant 0.9244 |
| Generic graph SSL | B3-DGI | 0.8624 | 0.8144 | 0.8619 | folded homogeneous graph + PubMedBERT, 300 epochs (re-run 2026-10-05; validation Acc 0.8638) |
| Generic graph SSL | B3-GraphMAE | 0.8410 | 0.7827 | 0.8399 | same as DGI (re-run 2026-10-05; validation Acc 0.8427) |

> Note: B1/B2 were run on 2026-10-03; the two B3 rows come from the **single run** of 2026-10-05
> (DGI 0.8624 / GraphMAE 0.8410, traceable to `result_B3_dgi.json` and `result_B3_graphmae.json`);
> they were not replicated and should be regarded as a single observation. Project history: DGI ran
> once on 2026-10-03 giving 0.8459, 1.65 points from the re-run value; because the artefacts of that
> first run were not archived, the paper no longer reports it as two independent runs. All three
> baseline groups generate their splits with the same stratified algorithm and seed as the main
> experiment (`test_ratio=0.1`, `val_ratio=0.1`, `split_seed=42`), and the resulting set sizes match
> the main experiment (97,641 / 10,837 / 12,041); the splits have been **verified member by member**
> against the main experiment (see `06_split_verification/split_verification_record.md`).
> `result_B1_bge.json` records `probe_C=4.0` with accuracy 0.8684 (the script writes 4 decimals via
> `round(..., 4)`, consistent with the run log; there is no precision discrepancy).

---

## SapBERT metric re-check (corrected)

The early `noGNN/SapBERT/test_metrics_noGNN.json` was byte-identical to the PubMedBERT file (same
md5) because SapBERT had not been re-run under the three-way protocol and the file had been
overwritten by the older PubMedBERT result. **This was corrected by a re-run on 2026-10-04**: the
current SapBERT three-way test results are Acc 0.8915 / Macro-F1 0.8533, and SapBERT, PubMedBERT
(0.8345 / 0.7793) and CODER (0.8947 / 0.8571) are now independent with distinct md5 values. The
earlier judgement that "the two models coincidentally agree" is void.
