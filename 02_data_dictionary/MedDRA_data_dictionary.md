# MedDRA Data Dictionary (for reproducing the experiments)

> The input to our experiments is a **flattened MedDRA hierarchy table**: each row corresponds to
> "one LLT term × one path to a SOC". A term that hangs on several paths appears in several rows —
> this is exactly the source of the multi-parent DAG structure modelled in this paper.
> The MedDRA source data are protected by the MSSO licence and cannot be redistributed publicly;
> this file describes fields and semantics only and contains no real MedDRA content.

## 1. Required fields

| Field | Type | Meaning | Use |
|---|---|---|---|
| `SOC_CODE` | int | System Organ Class code | Classification label (SOC level); graph construction |
| `SOC_NAME_EN` | str | System Organ Class name (English) | Node text feature |
| `HLGT_CODE` | int | High Level Group Term code | Graph construction |
| `HLGT_NAME_EN` | str | High Level Group Term name (English) | Node text feature |
| `HLT_CODE` | int | High Level Term code | Graph construction |
| `HLT_NAME_EN` | str | High Level Term name (English) | Node text feature |
| `PT_CODE` | int | Preferred Term code | Graph construction |
| `PT_NAME_EN` | str | Preferred Term name (English) | Node text feature |
| `LLT_CODE` | int | Lowest Level Term code | Graph construction |
| `LLT_NAME_EN` | str | Lowest Level Term name (English) | Node text feature |
| `PT_SOC_CODE` | int | **Official primary SOC** code of the PT | SOC label construction (PT level) |
| `PRIMARY_SOC_FG` | str (`Y`/`N`) | Whether this row is the term's **primary path** | Decides the "primary parent" in contrastive learning |

## 2. Row structure and the multi-parent mechanism

- Each row gives one complete path `LLT → PT → HLT → HLGT → SOC`;
- the processor establishes parent-child relations for **all rows** (`_add_relationship`), so a PT
  attached to different HLTs in different rows forms **multiple parents**;
- only rows with `PRIMARY_SOC_FG='Y'` are used to record the "primary parent" (`_primary_parent`);
  this serves the primary/secondary-parent decision in contrastive learning and does **not**
  participate in SOC label construction;
- SOC labels at the PT level come from `PT_SOC_CODE`; LLT inherits from its parent PT; HLT/HLGT are
  aggregated by majority vote over all child nodes;
- SOC-level nodes do **not** participate in training or evaluation (the 120,519 nodes reported in
  the paper are "non-SOC nodes").

## 3. Synthetic demo dataset

`make_synthetic_demo.py` generates `synthetic_meddra_demo.csv`:

- the five levels are complete (SOC / HLGT / HLT / PT / LLT), with the same fields and types as
  the table above;
- term names and codes are **entirely fictitious** (e.g. `Synthetic PT term 007 (demo)`) and contain
  no real MedDRA content;
- it deliberately includes **multi-parent PTs** (the same PT appears on two paths, one of which has
  `PRIMARY_SOC_FG='Y'`), so that graph construction, primary-parent determination and the
  contrastive-loss masks can be verified;
- it contains a few dozen nodes and runs in seconds on CPU; it is for **pipeline validation only**
  and yields no performance metrics.

```bash
python make_synthetic_demo.py                 # write the CSV (default: current directory)
python make_synthetic_demo.py --out my.csv    # specify the output path
```

To reproduce the real experiments, point the data path of the scripts to the **official MedDRA 29.0
table obtained under an MSSO licence**; all remaining steps (graph construction / training /
evaluation / diagnosis) reuse the scripts of this paper unchanged.
