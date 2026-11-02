# -*- coding: utf-8 -*-
r"""multiaxial_summary.py — 单轴 / 多轴测试节点上的模型表现（离线，无 GPU，仅 pandas）。

回答正文修订稿 §4.2 的占位符：模型（及 kNN）在多轴节点（n_SOC≥2）与单轴节点上的
Acc / Macro-F1，多个种子给 mean ± sd（ddof=1），以及错误中多轴节点的占比。

与 subgroup_performance.py 的区别：后者按"层级 × 1/2/3+"细分且每个 run 单独列表；
本脚本只分"单轴 / 多轴"两组（另附各层多轴 Acc），并把同一配置的多个种子聚合成一行，直接对应论文句子。

输入：每个 --group 给一个名字和若干路径；路径可以是 run 目录（读其中的 test_predictions_*.json，
      忽略 UNVERIFIED_ 前缀的文件）或单个预测 JSON。缺失路径只告警、不报错。
预测记录字段：node_code / level（大小写均可）/ true_soc_code / pred_soc_code。
  · SAGE 训练器（2026-10-23 之后）、diagnose_joint_subgroup.py --export-predictions、
    补丁版 train_b1_text_embed.py（kNN）导出的文件都满足这一口径。

用法：
  python multiaxial_summary.py --csv MEDDRA_DATA_29_0.csv \
    --group V1 <review>/V1_S43 <review>/V1_S44 <review>/V1_S45 <review>/V1_S46 \
               <analysis_out>/exported_predictions/test_predictions_SAGE__C1V1.json \
    --group kNN-BGE baseline/baseline_outputs/kNN_bge \
    --md MULTIAXIAL_SUMMARY.md --json MULTIAXIAL_SUMMARY.json
"""
import argparse
import glob
import io
import json
import os

from rule_baseline_soc import load, macro_f1
from subgroup_performance import build_meta

LEVELS = ["HLGT", "HLT", "PT", "LLT"]


def _pred_files(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return [p for p in sorted(glob.glob(os.path.join(path, "test_predictions_*.json")))
                if not os.path.basename(p).startswith("UNVERIFIED_")]
    return []


def _sd(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def run_metrics(records, meta):
    """单个 run：总体 / 单轴 / 多轴 / 各层多轴 的 Acc 与 Macro-F1，错误中多轴占比，随机基准。"""
    rows, n_unmatched = [], 0
    for r in records:
        level = str(r.get("level", "")).upper()
        try:
            code = int(r["node_code"])
        except (KeyError, TypeError, ValueError):
            n_unmatched += 1
            continue
        m = meta.get(level, {}).get(code)
        if m is None or r.get("true_soc_code") is None or r.get("pred_soc_code") is None:
            n_unmatched += 1
            continue
        rows.append((level, int(r["true_soc_code"]), int(r["pred_soc_code"]), int(m["n_soc"])))
    if not rows:
        raise ValueError("没有可匹配的记录（%d 条全部未匹配到层级元数据）" % n_unmatched)

    def block(sel):
        sub = [x for x in rows if sel(x)]
        if not sub:
            return {"n": 0, "acc": None, "macro_f1": None}
        t = [x[1] for x in sub]
        p = [x[2] for x in sub]
        return {"n": len(sub), "acc": sum(1 for a, b in zip(t, p) if a == b) / len(sub),
                "macro_f1": macro_f1(t, p)}

    out = {
        "n": len(rows),
        "n_unmatched": n_unmatched,
        "all": block(lambda x: True),
        "single": block(lambda x: x[3] == 1),
        "multi": block(lambda x: x[3] >= 2),
        "multi_by_level": {lv: block(lambda x, lv=lv: x[0] == lv and x[3] >= 2) for lv in LEVELS},
        "single_by_level": {lv: block(lambda x, lv=lv: x[0] == lv and x[3] == 1) for lv in LEVELS},
    }
    errs = [x for x in rows if x[1] != x[2]]
    out["n_err"] = len(errs)
    out["n_err_multi"] = sum(1 for x in errs if x[3] >= 2)
    out["err_share_multi"] = (out["n_err_multi"] / len(errs)) if errs else None
    multi = [x for x in rows if x[3] >= 2]
    out["random_among_candidates_multi"] = (sum(1.0 / x[3] for x in multi) / len(multi)) if multi else None
    return out


def aggregate(runs):
    """同一配置的多个 run → mean / sd / n（只对非 None 的值求）。"""
    def collect(getter):
        vals = []
        for r in runs:
            try:
                v = getter(r)
            except (KeyError, TypeError):
                v = None
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return {"n": len(vals), "mean": sum(vals) / len(vals), "sd": _sd(vals),
                "min": min(vals), "max": max(vals)}

    agg = {}
    for part in ("all", "single", "multi"):
        for met in ("acc", "macro_f1"):
            agg["%s_%s" % (part, met)] = collect(lambda r, part=part, met=met: r[part][met])
    for lv in LEVELS:
        agg["multi_%s_acc" % lv] = collect(lambda r, lv=lv: r["multi_by_level"][lv]["acc"])
    agg["err_share_multi"] = collect(lambda r: r["err_share_multi"])
    agg["n_err"] = collect(lambda r: r["n_err"])
    agg["n_err_multi"] = collect(lambda r: r["n_err_multi"])
    return agg


def fmt(s, pct=False):
    if not s:
        return "—"
    k = 100.0 if pct else 1.0
    if s["n"] >= 2:
        return "%.4f ± %.4f (n=%d)" % (s["mean"] * k, s["sd"] * k, s["n"]) if not pct else \
               "%.2f%% ± %.2f (n=%d)" % (s["mean"] * k, s["sd"] * k, s["n"])
    return ("%.4f (n=1)" % (s["mean"] * k)) if not pct else ("%.2f%% (n=1)" % (s["mean"] * k))


def fmt_count(s):
    if not s:
        return "—"
    return ("%.1f ± %.1f (n=%d)" % (s["mean"], s["sd"], s["n"])) if s["n"] >= 2 else ("%d" % round(s["mean"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "PATH"),
                    help="组名 + 若干 run 目录或预测 JSON；可重复给多组")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    df = load(a.csv)
    meta = build_meta(df)

    result = {"groups": {}}
    lines = ["# 单轴 / 多轴测试节点上的表现（离线复算）", "",
             "> 单轴 = 该节点在长表中只落在 1 个 SOC 下；多轴 = n_SOC ≥ 2。只评估主 SOC。",
             "> 多个种子给 mean ± sd（ddof=1）。`random` = 在每个多轴节点的候选 SOC 中均匀随机选一个的期望准确率。", ""]
    summary_rows = []
    for grp in a.group:
        name, paths = grp[0], grp[1:]
        files, missing = [], []
        for p in paths:
            fs = _pred_files(p)
            (files.extend(fs) if fs else missing.append(p))
        runs, per_run = [], {}
        for fp in files:
            try:
                recs = json.load(io.open(fp, encoding="utf-8"))
                rm = run_metrics(recs, meta)
            except Exception as e:
                print("[warn] %s: 跳过 %s（%s）" % (name, fp, e))
                continue
            runs.append(rm)
            per_run[fp] = rm
        if missing:
            print("[warn] %s: 以下路径不存在或没有预测文件，已跳过：" % name)
            for m in missing:
                print("        - %s" % m)
        if not runs:
            print("[warn] %s: 没有可用的预测文件" % name)
            result["groups"][name] = {"runs": {}, "agg": None, "missing": missing}
            continue
        ns = sorted({r["n"] for r in runs})
        if len(ns) > 1:
            print("[warn] %s: 各 run 的测试节点数不一致 %s（是否混入了不同划分 / 版本？）" % (name, ns))
        agg = aggregate(runs)
        result["groups"][name] = {"runs": per_run, "agg": agg, "missing": missing}
        r0 = runs[0]
        summary_rows.append((name, agg, r0))

        lines += ["## %s（%d 个 run）" % (name, len(runs)), ""]
        lines += ["| 子集 | n | Acc | Macro-F1 |", "| --- | --- | --- | --- |"]
        lines.append("| 全部 | %d | %s | %s |" % (r0["all"]["n"], fmt(agg["all_acc"]), fmt(agg["all_macro_f1"])))
        lines.append("| 单轴 | %d | %s | %s |" % (r0["single"]["n"], fmt(agg["single_acc"]), fmt(agg["single_macro_f1"])))
        lines.append("| 多轴 | %d | %s | %s |" % (r0["multi"]["n"], fmt(agg["multi_acc"]), fmt(agg["multi_macro_f1"])))
        for lv in LEVELS:
            if r0["multi_by_level"][lv]["n"]:
                lines.append("| 多轴 · %s | %d | %s | — |" % (lv, r0["multi_by_level"][lv]["n"], fmt(agg["multi_%s_acc" % lv])))
        lines += ["", "- 错误数：%s；其中多轴节点：%s；占比：%s" % (
            fmt_count(agg["n_err"]), fmt_count(agg["n_err_multi"]), fmt(agg["err_share_multi"], pct=True)),
            "- 多轴节点上随机选候选 SOC 的期望准确率：%.4f" % (r0["random_among_candidates_multi"] or 0.0), ""]
        for fp, rm in per_run.items():
            lines.append("  - `%s`：all %.4f / single %.4f / multi %.4f（n=%d，未匹配 %d）" % (
                fp, rm["all"]["acc"], rm["single"]["acc"] or 0, rm["multi"]["acc"] or 0, rm["n"], rm["n_unmatched"]))
        lines.append("")

    if summary_rows:
        lines += ["## 汇总（便于直接写入正文）", "",
                  "| 组 | 单轴 Acc | 多轴 Acc | 多轴 Macro-F1 | 错误中多轴占比 |", "| --- | --- | --- | --- | --- |"]
        for name, agg, _r0 in summary_rows:
            lines.append("| %s | %s | %s | %s | %s |" % (name, fmt(agg["single_acc"]), fmt(agg["multi_acc"]),
                                                     fmt(agg["multi_macro_f1"]), fmt(agg["err_share_multi"], pct=True)))
        lines.append("")
    md = "\n".join(lines)
    print(md)
    if a.md:
        io.open(a.md, "w", encoding="utf-8", newline="\n").write(md)
        print("written:", a.md)
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="\n").write(json.dumps(result, ensure_ascii=False, indent=1))
        print("written:", a.json)
    return result


if __name__ == "__main__":
    main()
