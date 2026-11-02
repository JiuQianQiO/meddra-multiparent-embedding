# -*- coding: utf-8 -*-
r"""rule_baseline_soc.py — E0 确定性 SOC 规则基线（离线，无 GPU，仅 pandas）。

目的：量化“给定 MedDRA 长表路径 / split_indices.json，primary-SOC 标签多大程度可被确定性规则复现”，
给 SOC 分类任务定级（是可见结构下的诊断任务，还是非平凡泛化任务）。

口径（均在 test 节点上评估）：
  R1 label_lookup : 复现论文标签构建函数（PT=PT_SOC_CODE；LLT 继承父 PT；HLT/HLGT 多数投票）。
  R2 topo_unique  : 只读层级路径（同节点的所有 SOC 祖先）；唯一祖先才预测，否则弃权。
  R3 topo_first   : 只读层级路径，多歧义取“首个 SOC 祖先”。
  R4 topo_majority: 只读层级路径，多歧义取“路径数量”多数 SOC。

用法：python rule_baseline_soc.py [--csv PATH] [--split PATH] [--md PATH]
"""
import argparse
import io
import json
import os
from collections import Counter

import pandas as pd

ROOT = r"D:\MedDRA模型文件\JBI"


def _int_series(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    for col in ["SOC_CODE", "HLGT_CODE", "HLT_CODE", "PT_CODE", "LLT_CODE", "PT_SOC_CODE"]:
        df[col] = _int_series(df[col])
    return df


def split_level_nodes(df):
    """返回与 trainer 一致的每层有序节点列表（按代码升序）。"""
    return {
        "HLGT": sorted(df["HLGT_CODE"].dropna().unique().tolist()),
        "HLT": sorted(df["HLT_CODE"].dropna().unique().tolist()),
        "PT": sorted(df["PT_CODE"].dropna().unique().tolist()),
        "LLT": sorted(df["LLT_CODE"].dropna().unique().tolist()),
    }


def build_labels(df):
    """复现 build_hierarchy 的 node_soc_map（按 level 存储，避免跨层同码歧义）。"""
    # PT 官方主 SOC（每个 PT_CODE 唯一）
    pt_soc = df.drop_duplicates("PT_CODE").dropna(subset=["PT_CODE"])
    pt_label = dict(zip(pt_soc["PT_CODE"], pt_soc["PT_SOC_CODE"]))

    # LLT 继承父 PT（LLT->PT 均唯一，取任一即可）
    llt_pt = df.drop_duplicates("LLT_CODE").dropna(subset=["LLT_CODE", "PT_CODE"])
    llt_label = {llt: pt_label.get(pt) for llt, pt in zip(llt_pt["LLT_CODE"], llt_pt["PT_CODE"])}

    # HLT 多数投票（子 PT 的 primary SOC）
    hlt_label = {}
    for hlt, g in df.dropna(subset=["HLT_CODE"]).groupby("HLT_CODE"):
        votes = [pt_label[c] for c in g["PT_CODE"].dropna().unique() if c in pt_label]
        if votes:
            hlt_label[hlt] = Counter(votes).most_common(1)[0][0]

    # HLGT 多数投票（子 HLT 的已给标签）
    hlgt_label = {}
    for hlgt, g in df.dropna(subset=["HLGT_CODE"]).groupby("HLGT_CODE"):
        votes = [hlt_label[c] for c in g["HLT_CODE"].dropna().unique() if c in hlt_label]
        if votes:
            hlgt_label[hlgt] = Counter(votes).most_common(1)[0][0]
    return {"PT": pt_label, "LLT": llt_label, "HLT": hlt_label, "HLGT": hlgt_label}


def build_soc_ancestors(df):
    """每题 (level, code) -> 有序 SOC 祖先列表（长表各行即一条路径）。"""
    out = {"HLGT": {}, "HLT": {}, "PT": {}, "LLT": {}}
    col = {"HLGT": "HLGT_CODE", "HLT": "HLT_CODE", "PT": "PT_CODE", "LLT": "LLT_CODE"}
    for level in out:
        for code, g in df.dropna(subset=[col[level]]).groupby(col[level]):
            out[level][code] = g["SOC_CODE"].dropna().tolist()
    return out


def macro_f1(y_true, y_pred):
    cats = sorted(set(y_true) | set(y_pred))
    f1s = []
    for c in cats:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t != c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != c and t == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def test_entries(split_path, level_nodes):
    """把 split 的 test 索引映射为 [(level, code), ...]（按 trainer 的 all_nodes 顺序）。"""
    sp = json.load(io.open(split_path, encoding="utf-8"))
    order = ["HLGT", "HLT", "PT", "LLT"]
    flat = [("".join(level), c) for level in order for c in level_nodes[level]]
    nodes_codes = [int(c) for c in sp["nodes"]]
    assert [c for _, c in flat] == nodes_codes, "split nodes 与有序层级节点不一致"
    return [flat[i] for i in sp["test"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "MEDDRA_DATA_29_0.csv"))
    ap.add_argument("--split", default=os.path.join(ROOT, "split_indices.json"))
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    df = load(a.csv)
    level_nodes = split_level_nodes(df)
    labels = build_labels(df)
    ancestors = build_soc_ancestors(df)
    entries = test_entries(a.split, level_nodes)

    def preds(fn):
        yt, yp = [], []
        for level, code in entries:
            yt.append(labels[level].get(code, -1))
            yp.append(fn(level, code))
        return yt, yp

    def r1(level, code):
        return labels[level].get(code, -1)

    def r2(level, code):
        a_ = ancestors[level].get(code, [])
        u = list(dict.fromkeys(a_))
        return u[0] if len(u) == 1 else -1

    def r3(level, code):
        a_ = ancestors[level].get(code, [])
        u = list(dict.fromkeys(a_))
        return u[0] if u else -1

    def r4(level, code):
        a_ = ancestors[level].get(code, [])
        return Counter(a_).most_common(1)[0][0] if a_ else -1

    lines = ["# E0 确定性 SOC 规则基线（离线复算）", "",
             "> 对照：SAGE C1V1 test Acc 0.9932 / Macro-F1 0.9892（12,041 节点）。", "",
             "> 口径：R1=标签字段规则；R2=纯拓扑·唯一祖先（多义弃权）；R3=纯拓扑·首祖先；R4=纯拓扑·路径多数。",
             "> 注：弃权类 R2 中约定 -1 为弃权，Acc 分母为全部 test 节点，以便与模型可比值。", ""]
    for name, fn in [("R1 label_lookup", r1), ("R2 topo_unique", r2),
                     ("R3 topo_first", r3), ("R4 topo_majority", r4)]:
        yt, yp = preds(fn)
        acc = sum(1 for t, p in zip(yt, yp) if t == p) / len(yt)
        mf1 = macro_f1(yt, yp)
        lines.append("## %s   Acc=%.4f  Macro-F1=%.4f" % (name, acc, mf1))
        lines.append("| 层级 | n | Acc | Macro-F1 |")
        lines.append("| --- | --- | --- | --- |")
        for level in ["HLGT", "HLT", "PT", "LLT"]:
            idx = [i for i, e in enumerate(entries) if e[0] == level]
            if not idx:
                continue
            t = [yt[i] for i in idx]
            p = [yp[i] for i in idx]
            acc_l = sum(1 for x, y in zip(t, p) if x == y) / len(t)
            m = macro_f1(t, p)
            lines.append("| %s | %d | %.4f | %.4f |" % (level, len(t), acc_l, m))
        lines.append("")
    md = "\n".join(lines)
    print(md)
    if a.md:
        io.open(a.md, "w", encoding="utf-8", newline="\n").write(md)
        print("\nwritten:", a.md)


if __name__ == "__main__":
    main()
