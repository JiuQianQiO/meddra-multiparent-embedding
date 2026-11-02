# -*- coding: utf-8 -*-
r"""subgroup_multiaxial.py — E4 多 SOC / 多父分层统计（离线，无 GPU，仅 pandas）。

把论文里混用的两个概念拆开：
  * multi-parent：某层级节点有 >1 个父节点（父层级列去重后 >1）；
  * multiaxial / multi-SOC：某层级节点在长表中落在 >1 个不同 SOC_CODE 下。

输出：全量与 test 两份口径的按层统计，供 3.1 / Abstract / 回复信精确引用。
用法：python subgroup_multiaxial.py [--csv PATH] [--split PATH] [--md PATH]
"""
import argparse
import io
import os
from collections import Counter

import pandas as pd

from rule_baseline_soc import load, split_level_nodes, test_entries

ROOT = r"D:\MedDRA模型文件\JBI"
LEVELS = ["HLGT", "HLT", "PT", "LLT"]
COL = {"HLGT": "HLGT_CODE", "HLT": "HLT_CODE", "PT": "PT_CODE", "LLT": "LLT_CODE"}
PARENT_COL = {"HLGT": "SOC_CODE", "HLT": "HLGT_CODE", "PT": "HLT_CODE", "LLT": "PT_CODE"}


def level_stats(df, level, selected_codes=None):
    """按层统计 multi-parent 与 multi-SOC；selected_codes 限定某 split 子集。"""
    c = COL[level]
    pc = PARENT_COL[level]
    sub = df.dropna(subset=[c])
    if selected_codes is not None:
        sub = sub[sub[c].isin(selected_codes)]
    if sub.empty:
        return None
    g = sub.groupby(c).agg(n_parent=(pc, "nunique"), n_soc=("SOC_CODE", "nunique"))
    n = len(g)
    mp = int((g.n_parent > 1).sum())
    ms = int((g.n_soc > 1).sum())
    soc_dist = Counter(g.n_soc.tolist())
    parent_dist = Counter(g.n_parent.tolist())
    return {
        "level": level, "n": n,
        "multi_parent": mp, "multi_parent_pct": 100 * mp / n,
        "multi_soc": ms, "multi_soc_pct": 100 * ms / n,
        "max_parent": int(g.n_parent.max()),
        "max_soc": int(g.n_soc.max()),
        "soc_dist": dict(sorted(soc_dist.items())),
        "parent_dist": dict(sorted(parent_dist.items())),
    }


def render(rows, title):
    lines = ["## %s" % title, "",
             "| 层级 | n | 多父（>1 父） | % | 多 SOC（>1 SOC） | % | 父数上限 | SOC 数上限 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append("| %s | %d | %d | %.2f%% | %d | %.2f%% | %d | %d |" % (
            r["level"], r["n"], r["multi_parent"], r["multi_parent_pct"],
            r["multi_soc"], r["multi_soc_pct"], r["max_parent"], r["max_soc"]))
    lines.append("")
    for r in rows:
        lines.append("- %s `n_SOC` 分布=%s；`n_parent` 分布=%s" % (
            r["level"], r["soc_dist"], r["parent_dist"]))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "MEDDRA_DATA_29_0.csv"))
    ap.add_argument("--split", default=os.path.join(ROOT, "split_indices.json"))
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    df = load(a.csv)
    level_nodes = split_level_nodes(df)
    full = [level_stats(df, lv) for lv in LEVELS]

    entries = test_entries(a.split, level_nodes)
    test_codes = {lv: [c for l, c in entries if l == lv] for lv in LEVELS}
    test_rows = [level_stats(df, lv, set(test_codes.get(lv, []))) for lv in LEVELS]
    test_rows = [r for r in test_rows if r]

    lines = ["# E4 多父 / 多 SOC 分层统计（离线复算）", "",
             "> 数据：`%s`；test 口径取自 `%s`（n_test=12,041）。" % (a.csv, a.split), ""]
    lines.append(render(full, "全量（120,519 非 SOC 节点）"))
    lines.append(render(test_rows, "独立测试集（12,041 节点）"))
    md = "\n".join(lines)
    print(md)
    if a.md:
        io.open(a.md, "w", encoding="utf-8", newline="\n").write(md)
        print("\nwritten:", a.md)


if __name__ == "__main__":
    main()
