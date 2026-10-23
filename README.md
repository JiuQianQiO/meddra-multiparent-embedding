# MedDRA Multi-Parent DAG Term Embeddings — Reproduction Package

Reproduction code and experimental artefacts for the manuscript submitted to the
*Journal of Biomedical Informatics* (JBI):

> **Medical Term Embedding for MedDRA Multi-Parent DAG: A Heterogeneous Graph Approach with
> Multi-Level Contrastive Learning**

The method embeds MedDRA terms by combining term-name semantics (PubMedBERT) with the
five-level multi-parent hierarchy, using a heterogeneous graph encoder and a multi-level
weighted contrastive loss with positive–negative mutual exclusion.

---

## What is here, and what is not

| Included | Not included (and why) |
|---|---|
| Code for graph construction, training, evaluation, diagnosis, baselines and figures | **MedDRA source data** — distributed under MSSO licence; must not be redistributed |
| Field-description document + **synthetic demo dataset** (no licence needed) | Manuscript files, cover letter, reviewer correspondence |
| Consolidated result tables and per-run metrics / histories / diagnostics | Model weights (1.4 GB) — see "Artefacts" below |
| Resource ledger (parameters, memory, peak GPU memory, per-epoch time) | Cached graph/feature tensors |
| Train/validation/test split verification record | |

> **MedDRA licence.** MedDRA is distributed by the MSSO. Academic users must obtain their own
> subscription or MSSO approval; commercial users must purchase a licence. The raw data are
> **not** redistributed here in any form. Everything needed to reproduce the *code path* is
> included via the synthetic demo dataset.

---

## Repository layout

| Directory | Contents |
|---|---|
| `01_code/` | `Meddra_full_trainer_SAGE.py` (main method), `meddra_full_trainer_GAT.py`, `meddra_full_trainer_noGNN.py`; inductive variants `Meddra_inductive_{SAGE,GAT,noGNN}.py`; seed re-runs `train_C7_seed43.py`, `train_V3_seed43.py`; `diagnose_joint_subgroup.py`; `baseline/` = text-embedding (B1), KGE (B2) and graph-SSL (B3) baselines |
| `02_data_dictionary/` | `MedDRA_data_dictionary.md` (flat-table structure, required fields, how the multi-parent DAG arises, primary-parent rule), `synthetic_meddra_demo.csv`, `make_synthetic_demo.py` |
| `04_results/` | `results_summary_5tables.md` — consolidated result tables for cross-checking against the manuscript |
| `05_resource_ledger/` | `platform_resources.csv` — parameters, memory, peak GPU memory and per-epoch time for 22 configurations |
| `06_split_verification/` | `verify_split_alignment.py`, `test_split_equiv.py`, `split_verification_record.md` — evidence that every run shares one identical split |
| `07_run_artifacts/` | Per-run `test_metrics_*.json`, `joint_subgroup_diag.json`, `RUN_META.json` (provenance), training histories, and external-baseline `result_B*.json`. Every number in the manuscript traces to these files. |

---

## Environment

Used for the reported results (also stated in Section 3.5 and Appendix B.6 of the manuscript):

- **OS** Ubuntu 22.04 · **Python** 3.12 · **PyTorch** 2.14.0
- PyTorch Geometric, Transformers (PubMedBERT / SapBERT / CODER), scikit-learn, pandas, numpy, matplotlib

Exact pins are in `requirements.txt`. Packages marked `>=` had no patch version recorded in
the run metadata and are therefore given as minimums.

---

## Reproducing the results

1. **Obtain the data.** Get `MEDDRA_DATA_29_0.csv` (and `MEDDRA_DATA_29_1.csv` for the
   cross-version check) under your own MSSO licence, then set:
   ```bash
   export MEDDRA_CSV_PATH=/your/path/MEDDRA_DATA_29_0.csv
   export MEDDRA_OUTPUT_ROOT=/your/output/dir
   ```
   To exercise the pipeline **without** a licence, use the synthetic demo:
   ```bash
   export MEDDRA_CSV_PATH=02_data_dictionary/synthetic_meddra_demo.csv
   python 01_code/Meddra_full_trainer_SAGE.py          # smoke test only
   ```
2. **Main method:** `python 01_code/Meddra_full_trainer_SAGE.py`
3. **Comparisons / ablations:** `meddra_full_trainer_GAT.py`, `meddra_full_trainer_noGNN.py`,
   `train_C7_seed43.py`, `train_V3_seed43.py`
4. **Inductive setting:** `Meddra_inductive_{SAGE,GAT,noGNN}.py`
   (both protocols are selected by experiment name — see each script's header)
5. **External baselines:** `baseline/train_b1_text_embed.py`, `baseline/train_b2_kge.py`,
   `baseline/train_b3_graph_ssl.py`
6. **Attribution diagnosis:** `python 01_code/diagnose_joint_subgroup.py`

Outputs go to `MEDDRA_OUTPUT_ROOT` (defaults to the working directory). Each run writes its
own `RUN_META.json`, which records the script hash, argv and resolved paths, so a result can
always be traced back to the code that produced it.

**Split integrity.** The split is exported once to `split_indices.json` and every script
verifies it member-by-member rather than by size; see `06_split_verification/`.

---

## Artefacts

`07_run_artifacts/` contains every JSON/CSV product the manuscript cites (approx. 8.7 MB):
metrics, training histories, joint-diagnosis output, baseline results and provenance records.

Deliberately **excluded** to keep the archive small and avoid redundancy:

- **Node embedding CSVs** (approx. 148 MB) — regenerated from a checkpoint by the same scripts.
- **Model weights** (`.pt`/`.pth`, approx. 1.4 GB) and **cached graph/feature tensors**
  (approx. 353 MB) — reproduced by re-running training with the settings in the manuscript.

If a specific checkpoint is needed, re-run the corresponding configuration; training is
deterministic given `split_seed=42` and the recorded `train_seed`.

---

## Licence

- **Code:** MIT (see `LICENSE`).
- **Documentation and the synthetic demo dataset:** CC BY 4.0.
- **MedDRA data:** not included; subject to MSSO licensing.

---

## Citing

If you use this code, please cite the JBI article and the archived artefact set:

- Code repository: https://github.com/JiuQianQiO/meddra-multiparent-embedding
- Long-term archive: `<Zenodo DOI — to be added after the first release>`
