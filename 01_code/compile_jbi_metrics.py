# -*- coding: utf-8 -*-
r"""compile_jbi_metrics.py

汇总 D:\MedDRA模型文件\JBI 下已有的 JBI 投稿实验结果（test_metrics_*.json），
出一张可复核的指标表（test Acc / Macro-F1 / 每层 Macro-F1 / selected_epoch / 测试节点数）。

用途：
  1. 核对论文主表与实测产物是否一致（官方 V1 应为 0.9932 / 0.9892）；
  2. E1-E3 跑完后，用同一脚本把 REVIEW 目录的新结果并入同口径表。

用法：
  python compile_jbi_metrics.py
  python compile_jbi_metrics.py --md <out.md>
"""
import argparse
import io
import json
import os

ROOT = r"D:\MedDRA模型文件\JBI"

COLS = ["family", "run", "acc", "macro_f1", "weighted_f1",
        "hlgt", "hlt", "pt", "llt", "epoch", "n_test"]


def scan(root):
    rows = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not (fn.startswith("test_metrics_") and fn.endswith(".json")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, encoding="utf-8") as f:
                    tm = json.load(f)
            except Exception as e:
                print("[warn] skip %s: %s" % (p, e))
                continue
            rel = os.path.relpath(dirpath, root)
            parts = [x for x in rel.split(os.sep) if x]
            family = parts[0] if parts else ""
            run = os.path.basename(dirpath)
            n = tm.get("num_test_nodes") or tm.get("llt_n", 0)
            rows.append({
                "family": family,
                "run": run,
                "acc": tm.get("accuracy"),
                "macro_f1": tm.get("macro_f1"),
                "weighted_f1": tm.get("weighted_f1"),
                "hlgt": tm.get("hlgt_macro_f1"),
                "hlt": tm.get("hlt_macro_f1"),
                "pt": tm.get("pt_macro_f1"),
                "llt": tm.get("llt_macro_f1"),
                "epoch": tm.get("selected_epoch"),
                "n_test": n,
            })
    rows.sort(key=lambda r: (r["family"].lower(), r["run"].lower()))
    return rows


def fmt(v, nd=4):
    if v is None:
        return "-"
    try:
        return ("%0.*f" % (nd, float(v)))
    except Exception:
        return str(v)


def render_md(rows):
    out = []
    out.append("# JBI 投稿实测指标汇总")
    out.append("")
    out.append("> 来源：`%s` 下全部 `test_metrics_*.json`（独立测试集，三分协议）。" % ROOT)
    out.append("> 每层列为该层**单独计算的 Macro-F1**（非全文总 Macro-F1）。")
    out.append("")
    out.append("| 目录 | 运行 | Acc | Macro-F1 | Weighted-F1 | HLGT | HLT | PT | LLT | epoch | n_test |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["family"], r["run"], fmt(r["acc"]), fmt(r["macro_f1"]), fmt(r["weighted_f1"]),
            fmt(r["hlgt"]), fmt(r["hlt"]), fmt(r["pt"]), fmt(r["llt"]),
            r["epoch"] if r["epoch"] is not None else "-", r["n_test"]))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--md", default=None, help="写入 Markdown 文件路径（UTF-8 无 BOM）")
    ap.add_argument("--csv", default=None, help="写入 CSV 文件路径")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        raise SystemExit("目录不存在：%s" % a.root)

    rows = scan(a.root)
    md = render_md(rows)

    if a.md:
        with io.open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        print("written markdown:", a.md)
    if a.csv:
        import csv
        with io.open(a.csv, "w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for r in rows:
                w.writerow([r[c] for c in COLS])
        print("written csv:", a.csv)

    print(md)


if __name__ == "__main__":
    main()
