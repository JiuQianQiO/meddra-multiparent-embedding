# -*- coding: utf-8 -*-
r"""subgroup_performance.py — 多父 / 多SOC子群的“模型性能”分层（离线，无 GPU）。

在训练结束生成 `test_predictions_<tag>.json` 后运行；不加载 checkpoint，只做 pandas + 指标聚合。
它回答审稿报告 E2/P0：模型对单父 vs 多父、单轴 vs 多轴节点是否表现不同。

用法：
    python subgroup_performance.py
      --root D:\\MedDRA模型文件\\JBI\\REVIEW
      --csv  D:\\MedDRA模型文件\\JBI\\MEDDRA_DATA_29_0.csv
      --split D:\\MedDRA模型文件\\JBI\\split_indices.json
      --md   ..\\paper\\JBI\\E2b_多父子群性能分层.md

输出：
    每个 run 目录 subgroup_performance_<dir>.json；聚合表写 --md。
"""
import argparse
import glob
import io
import json
import os

import pandas as pd

from rule_baseline_soc import load, split_level_nodes, macro_f1

ROOT = r"D:\MedDRA模型文件\JBI"
LEVELS = ["HLGT", "HLT", "PT", "LLT"]
COL = {"HLGT": "HLGT_CODE", "HLT": "HLT_CODE", "PT": "PT_CODE", "LLT": "LLT_CODE"}
PARENT_COL = {"HLGT": "SOC_CODE", "HLT": "HLGT_CODE", "PT": "HLT_CODE", "LLT": "PT_CODE"}


def build_meta(df):
    meta = {}
    for level in LEVELS:
        c = COL[level]
        pc = PARENT_COL[level]
        meta[level] = {}
        sub = df.dropna(subset=[c, pc, "SOC_CODE"])
        for code, g in sub.groupby(c):
            meta[level][int(code)] = {
                "n_parent": int(g[pc].nunique()),
                "n_soc": int(g["SOC_CODE"].nunique()),
            }
    return meta


def bucket_label(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def group_metrics(records, meta):
    buckets = {"parent": {}, "soc": {}}
    n_unmatched = 0
    for r in records:
        # 训练器 save_test_predictions() 写的 level 是小写节点类型（"pt"/"llt"…），
        # meta 的键是大写层级（"PT"/"LLT"…）。必须统一大小写，否则全部记录查不到 meta、
        # 被静默跳过，子群表变成空表（2026-10-29 核查发现）。
        code = int(r["node_code"])
        level = str(r["level"]).upper()
        m = meta.get(level, {}).get(code)
        if m is None:
            n_unmatched += 1
            continue
        for kind, key, val in (("parent", "n_parent", m["n_parent"]),
                               ("soc", "n_soc", m["n_soc"])):
            b = bucket_label(val)
            cell = buckets[kind].setdefault((level, b), {"truth": [], "pred": []})
            if r.get("true_soc_code") is not None and r.get("pred_soc_code") is not None:
                cell["truth"].append(int(r["true_soc_code"]))
                cell["pred"].append(int(r["pred_soc_code"]))
    out = {"parent": {}, "soc": {}}
    for kind in out:
        for (level, b), cell in sorted(buckets[kind].items()):
            t, p = cell["truth"], cell["pred"]
            n = len(t)
            acc = sum(1 for x, y in zip(t, p) if x == y) / n if n else None
            mf1 = macro_f1(t, p) if n else None
            out[kind][f"{level}:{b}"] = {"n": n, "acc": acc, "macro_f1": mf1}
    if records and n_unmatched == len(records):
        # 自检：一条都没匹配上说明 level / node_code 口径又不一致了，宁可失败也不出空表
        raise SystemExit("subgroup_performance: %d 条预测记录全部未匹配到层级元数据"
                         "（示例 level=%r, node_code=%r）；请检查 test_predictions 的字段口径。"
                         % (len(records), records[0].get("level"), records[0].get("node_code")))
    if n_unmatched:
        print("[warn] %d / %d 条预测记录未匹配到层级元数据，已跳过" % (n_unmatched, len(records)))
    return out


def render(rows, title):
    lines = [
        "## %s" % title,
        "",
        "| kind | level-bucket | n | Acc | Macro-F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append("| %(kind)s | %(key)s | %(n)s | %(acc)s | %(mf1)s |" % r)
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "REVIEW"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "MEDDRA_DATA_29_0.csv"))
    ap.add_argument("--split", default=os.path.join(ROOT, "split_indices.json"))
    ap.add_argument("--md", default="")
    ap.add_argument("--only", default="", help="只处理逗号分隔的 run 目录名")
    a = ap.parse_args()

    df = load(a.csv)
    meta = build_meta(df)
    run_dirs = sorted(d for d in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(d))
    if a.only:
        keep = set(x.strip().strip('"').strip("'") for x in a.only.split(",") if x.strip())
        run_dirs = [d for d in run_dirs if os.path.basename(d) in keep]

    lines = ["# E2b 多父 / 多SOC子句的模型性能分层（离线复算）", "",
             "> 数据：%s" % a.root, "> bucket：parent 与 soc 均按 1 / 2 / 3+。"]

    processed = 0
    skipped = []
    for d in run_dirs:
        pfiles = sorted(glob.glob(os.path.join(d, "test_predictions_*.json")))
        if not pfiles:
            skipped.append(os.path.basename(d))
            continue
        name = os.path.basename(d)
        try:
            with open(pfiles[0], encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            lines.append("- %s: 预测文件读取失败 (%s)" % (name, e))
            continue
        perf = group_metrics(records, meta)
        out_path = os.path.join(d, "subgroup_performance_%s.json" % name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(perf, f, ensure_ascii=False, indent=1)

        rows = []
        for kind in ("parent", "soc"):
            for key, v in sorted(perf[kind].items()):
                if v["n"]:
                    rows.append({
                        "kind": kind,
                        "key": key,
                        "n": v["n"],
                        "acc": None if v["acc"] is None else "%.4f" % v["acc"],
                        "mf1": None if v["macro_f1"] is None else "%.4f" % v["macro_f1"],
                    })
        lines.append("## %s" % name)
        lines.append("")
        lines.append(render(rows, "%s · subgroup performance" % name))
        processed += 1

    lines.append("")
    lines.append("已处理 %d 个 run 目录。" % processed)
    if skipped:
        lines.append("")
        lines.append("缺少 `test_predictions_*.json` 而被跳过的目录（GAT/noGNN 不产该文件属预期；如需覆盖请用 diagnose_joint_subgroup.py）：")
        for s in skipped:
            lines.append("- %s" % s)
    md = "\n".join(lines)
    print(md)
    if a.md:
        io.open(a.md, "w", encoding="utf-8", newline="\n").write(md)
        print("\nwritten:", a.md)


if __name__ == "__main__":
    main()
