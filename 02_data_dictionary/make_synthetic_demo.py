# -*- coding: utf-8 -*-
"""
Generate a fully synthetic miniature MedDRA-style dataset (for pipeline validation; contains no real MedDRA content).

Purpose: allow readers without a MedDRA subscription to run the graph construction / training / evaluation pipeline and verify the implementation.
See `MedDRA_data_dictionary.md` in the same directory for fields and semantics.

Output: a flat table with one row per LLT x path to SOC; multi-parent PTs are deliberately included.

Usage:
    python make_synthetic_demo.py [--out synthetic_meddra_demo.csv]
"""
from __future__ import annotations

import argparse
import csv


def build_rows():
    """Build a small five-level hierarchy:
    2 SOCs -> 2 HLGTs each -> 2 HLTs each -> 2 PTs each -> 2 LLTs each,
    and additionally attach 2 PTs to a second HLT (forming a multi-parent DAG).
    """
    rows = []
    code = {"SOC": 10000000, "HLGT": 20000000, "HLT": 30000000, "PT": 40000000, "LLT": 50000000}

    def new_code(level):
        code[level] += 1
        return code[level]

    # ---- hierarchy skeleton ----
    socs, hlgts, hlts, pts = [], [], [], []
    for i in range(2):
        s = new_code("SOC")
        socs.append((s, f"Synthetic SOC term {i + 1:02d} (demo)"))
        for j in range(2):
            g = new_code("HLGT")
            hlgts.append((g, f"Synthetic HLGT term {len(hlgts) + 1:02d} (demo)", s))
            for k in range(2):
                h = new_code("HLT")
                hlts.append((h, f"Synthetic HLT term {len(hlts) + 1:02d} (demo)", g, s))
                for m in range(2):
                    p = new_code("PT")
                    pts.append((p, f"Synthetic PT term {len(pts) + 1:02d} (demo)",
                                h, g, s))

    def name_of(items, c):
        for it in items:
            if it[0] == c:
                return it[1]
        return ""

    def path_of_pt(p_code):
        for p, _n, h, g, s in pts:
            if p == p_code:
                return h, g, s
        raise KeyError(p_code)

    # ---- add one extra path to each of the first two PTs (multi-parent) ----
    extra_parent = {}
    hlt_codes = [h for h, _n, _g, _s in hlts]
    for p_code, _n, _h, _g, _s in list(pts)[:2]:
        for alt_h in hlt_codes:
            if alt_h != path_of_pt(p_code)[0]:
                extra_parent[p_code] = alt_h
                break

    def hlt_path(h):
        for hh, _n, g, s in hlts:
            if hh == h:
                return g, s
        raise KeyError(h)

    # ---- generate one row per LLT ----
    for p, p_name, h, g, s in pts:
        primary_soc = s
        for i in range(2):
            l = new_code("LLT")
            l_name = f"Synthetic LLT term {len(rows) + 1:03d} (demo)"
            soc_name = name_of(socs, s)
            rows.append(dict(
                SOC_CODE=s, SOC_NAME_EN=soc_name,
                HLGT_CODE=g, HLGT_NAME_EN=name_of(hlgts, g),
                HLT_CODE=h, HLT_NAME_EN=name_of(hlts, h),
                PT_CODE=p, PT_NAME_EN=p_name,
                LLT_CODE=l, LLT_NAME_EN=l_name,
                PT_SOC_CODE=primary_soc, PRIMARY_SOC_FG="Y",
            ))
        # multi-parent PT: add one more row for the same LLTs (secondary path, PRIMARY_SOC_FG='N')
        if p in extra_parent:
            alt_h = extra_parent[p]
            alt_g, alt_s = hlt_path(alt_h)
            for r in [x for x in rows if x["PT_CODE"] == p][:2]:
                rows.append(dict(
                    SOC_CODE=alt_s, SOC_NAME_EN=name_of(socs, alt_s),
                    HLGT_CODE=alt_g, HLGT_NAME_EN=name_of(hlgts, alt_g),
                    HLT_CODE=alt_h, HLT_NAME_EN=name_of(hlts, alt_h),
                    PT_CODE=p, PT_NAME_EN=p_name,
                    LLT_CODE=r["LLT_CODE"], LLT_NAME_EN=r["LLT_NAME_EN"],
                    PT_SOC_CODE=primary_soc, PRIMARY_SOC_FG="N",
                ))

    fields = ["SOC_CODE", "SOC_NAME_EN", "HLGT_CODE", "HLGT_NAME_EN",
              "HLT_CODE", "HLT_NAME_EN", "PT_CODE", "PT_NAME_EN",
              "LLT_CODE", "LLT_NAME_EN", "PT_SOC_CODE", "PRIMARY_SOC_FG"]
    return rows, fields


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic miniature MedDRA-style demo dataset")
    ap.add_argument("--out", default="synthetic_meddra_demo.csv")
    args = ap.parse_args(argv)

    rows, fields = build_rows()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_soc = len({r["SOC_CODE"] for r in rows})
    n_pt = len({r["PT_CODE"] for r in rows})
    multi = [p for p in {r["PT_CODE"] for r in rows}
             if len({r["HLT_CODE"] for r in rows if r["PT_CODE"] == p}) > 1]
    print(f"Written: {args.out}")
    print(f"  rows={len(rows)}  SOC={n_soc}  PT={n_pt}  multi-parent PT={len(multi)} -> {multi}")
    print("  fields:", ", ".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
