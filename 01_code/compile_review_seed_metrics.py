# -*- coding: utf-8 -*-
"""compile_review_seed_metrics.py

汇总 run_review_experiments_full.py 产生的多种子结果：
把 run dir 名中的 `_S<seed>` 归一化为配置组，输出 n/mean/sd/min/max 与各种子单值。

用法：
  python compile_review_seed_metrics.py \
    --root /mnt/workspace/JBI/data/jbi_runs_review \
    --md   /mnt/workspace/JBI/data/jbi_runs_review/REVIEW_SEED_metrics.md \
    --csv  /mnt/workspace/JBI/data/jbi_runs_review/REVIEW_SEED_metrics.csv
"""
import argparse
import csv
import io
import json
import os
import re

COLS = ["group", "n", "acc_mean", "acc_sd", "acc_min", "acc_max",
        "macro_f1_mean", "macro_f1_sd", "macro_f1_min", "macro_f1_max",
        "hlgt", "hlt", "pt", "llt", "epoch", "n_test", "seeds"]

# 汇总的指标：总体 + 分层 Macro-F1 + 分层 Acc。
# 分层 Acc 用于 SOC 控制实验：3 层 GNN 下 LLT（占测试集约 75%）在结构上看不到 SOC 节点，
# 只看总体指标会把 HLGT/HLT/PT 上的效应稀释掉，因此 SOC 控制必须按层报告。
METRICS = ("acc", "macro_f1", "hlgt", "hlt", "pt", "llt",
           "hlgt_acc", "hlt_acc", "pt_acc", "llt_acc")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row(run, group, tm):
    """把一个 test_metrics_*.json 转成一行（scan 与 scan_base 共用，避免两处口径不一致）。"""
    return {
        "run": run,
        "group": group,
        "acc": _num(tm.get("accuracy")),
        "macro_f1": _num(tm.get("macro_f1")),
        "hlgt": _num(tm.get("hlgt_macro_f1")),
        "hlt": _num(tm.get("hlt_macro_f1")),
        "pt": _num(tm.get("pt_macro_f1")),
        "llt": _num(tm.get("llt_macro_f1")),
        "hlgt_acc": _num(tm.get("hlgt_acc")),
        "hlt_acc": _num(tm.get("hlt_acc")),
        "pt_acc": _num(tm.get("pt_acc")),
        "llt_acc": _num(tm.get("llt_acc")),
        "epoch": tm.get("selected_epoch"),
        "n_test": tm.get("num_test_nodes") or tm.get("llt_n", 0),
    }


def group_of(run_name):
    """V1_S43 -> V1；E1A_S42 -> E1A；TAU010 -> TAU010。"""
    return re.sub(r"_S\d+$", "", run_name)


def scan(root):
    rows = []
    for run in sorted(os.listdir(root)):
        run_dir = os.path.join(root, run)
        if not os.path.isdir(run_dir):
            continue
        files = [f for f in os.listdir(run_dir)
                 if f.startswith("test_metrics_") and f.endswith(".json")]
        if not files:
            continue
        fp = os.path.join(run_dir, sorted(files)[0])
        try:
            with io.open(fp, encoding="utf-8") as f:
                tm = json.load(f)
        except Exception as e:
            print("[warn] skip %s: %s" % (fp, e))
            continue
        rows.append(_row(run, group_of(run), tm))
    return rows


# 主结果根目录里的既有观察：(配置组, 训练种子, 候选相对目录...)。
# 候选目录按顺序取第一个存在的（本地 D:\ 与服务器布局不完全相同）。
# 2026-10-29：补入 C7 的 seed-43（train_C7_seed43.py 写 <OUTPUT_ROOT>/C7-seed43），
#   与 review 根里的 C7_S44/45/46 合起来 C7 才是 n=5（之前实际只有 {42,44,45,46}）。
#   V3 的 seed-43 不在此并入：P1 已重跑 V3_S43，若两边都并入会把 seed 43 算两次。
BASE_ENTRIES = [
    # 本地 D:\ 布局为 SAGE/<实验>；服务器第一轮按 RUNBOOK.md 用启动器模式跑时为 <OUTPUT_ROOT>/<实验>。
    # 两种都列为候选，取第一个存在的目录（同一组同一种子只会取一份）。
    ("V1", "42", ["SAGE/C1V1", "SAGE/V1C1", "SAGE/V1", "V1C1", "V1"]),
    ("V2", "42", ["SAGE/V2", "V2"]),
    ("V3", "42", ["SAGE/V3", "V3"]),
    ("V4", "42", ["SAGE/V4", "V4"]),
    ("C2", "42", ["SAGE/C2", "C2"]),
    ("C3", "42", ["SAGE/C3", "C3"]),
    ("C4", "42", ["SAGE/C4", "C4"]),
    ("C5", "42", ["SAGE/C5", "C5"]),
    ("C6", "42", ["SAGE/C6", "C6"]),
    ("C7", "42", ["SAGE/C7", "C7"]),
    ("C7", "43", ["C7-seed43", "SAGE/C7-seed43", "SAGE/C7_seed43"]),
    ("NGPB", "42", ["NOGNN/PubMedBERT", "NOGNN/PubMedBert", "PubMedBert", "PubMedBERT"]),
    ("NGSB", "42", ["NOGNN/SapBERT", "SapBERT"]),
    ("NGCODER", "42", ["NOGNN/CODER", "CODER"]),
    ("GAT", "42", ["GAT"]),
]


def scan_base(base_root):
    """从主结果根目录补入既有种子观察（seed 42 及 C7 的 seed 43），避免多种子汇总少观察。"""
    rows = []
    if not base_root or not os.path.isdir(base_root):
        if base_root:
            print("[warn] --base-root 不存在：%s（所有组都会缺少 seed-42 观察）" % base_root)
        return rows
    missing = []
    for group, seed, cands in BASE_ENTRIES:
        run_dir = None
        for rel in cands:
            p = os.path.join(base_root, *rel.split("/"))
            if os.path.isdir(p):
                run_dir, hit = p, rel
                break
        if run_dir is None:
            missing.append("%s seed=%s（候选：%s）" % (group, seed, " | ".join(cands)))
            continue
        files = sorted(f for f in os.listdir(run_dir)
                       if f.startswith("test_metrics_") and f.endswith(".json"))
        if not files:
            missing.append("%s seed=%s：%s 下无 test_metrics_*.json" % (group, seed, hit))
            continue
        fp = os.path.join(run_dir, files[0])
        try:
            with io.open(fp, encoding="utf-8") as f:
                tm = json.load(f)
        except Exception as e:
            print("[warn] skip %s: %s" % (fp, e))
            continue
        rows.append(_row(hit.replace("/", "__") + "_BASE%s" % seed, group, tm))
    if missing:
        print("[warn] 以下基准观察缺失，对应配置组的 n 会偏小：")
        for m in missing:
            print("        - %s" % m)
    return rows


def check_duplicate_seeds(rows):
    """同一配置组里同一种子出现两次（如 base 与 review 都有 seed 43）时告警。"""
    seen = {}
    for r in rows:
        m = re.search(r"_S(\d+)$", r["run"]) or re.search(r"_BASE(\d+)$", r["run"])
        if not m:
            continue
        seen.setdefault((r["group"], m.group(1)), []).append(r["run"])
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    for (g, sd), runs in sorted(dups.items()):
        print("[warn] 组 %s 的 seed %s 出现 %d 次：%s —— 请只保留一个，否则 n 与 sd 失真"
              % (g, sd, len(runs), ", ".join(runs)))
    return dups


def agg(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    out = []
    for group in sorted(groups):
        rs = sorted(groups[group], key=lambda x: x["run"])
        for metric in METRICS:
            vals = [r[metric] for r in rs if r[metric] is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                # 样本标准差（ddof=1）：论文写 mean ± sd，n 较小时（如 3/5）总体公式会系统性偏小。
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                sd = var ** 0.5
            else:
                sd = 0.0
            r0 = rs[0]
            out.append({
                "group_key": group,
                "group": group if metric == "acc" else "",
                "metric": metric,
                "n": len(vals),
                "mean": mean,
                "sd": sd,
                "min": min(vals),
                "max": max(vals),
                "seeds": ", ".join(r["run"] for r in rs),
                "n_test": r0["n_test"],
            })
    return out


def render_md(agg_rows):
    lines = ["# Review multi-seed metrics (mean ± sd)", "",
             "> metric：acc / macro_f1 为总体；hlgt/hlt/pt/llt 为分层 Macro-F1；*_acc 为分层 Acc。"
             "SOC 控制请按 HLGT/HLT/PT 分层解读（3 层 GNN 下 LLT 看不到 SOC）。sd 为样本标准差（ddof=1）。", "",
             "| group | metric | n | mean | sd | min | max | n_test | seeds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in agg_rows:
        lines.append("| %s | %s | %d | %.4f | %.4f | %.4f | %.4f | %s | %s |" % (
            r["group"], r["metric"], r["n"], r["mean"], r["sd"],
            r["min"], r["max"], r["n_test"], r["seeds"]))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("MEDDRA_REVIEW_ROOT", ""))
    ap.add_argument("--base-root", default=os.environ.get("MEDDRA_OUTPUT_ROOT", ""),
                    help="主结果根目录，用于补入 seed 42")
    ap.add_argument("--md", default="")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    if not a.root or not os.path.isdir(a.root):
        raise SystemExit("目录不存在，请用 --root 指定 review 结果根目录")

    rows = scan(a.root) + scan_base(a.base_root)
    check_duplicate_seeds(rows)
    agg_rows = agg(rows)
    md = render_md(agg_rows)
    print(md)

    if a.md:
        with io.open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        print("written:", a.md)
    if a.csv:
        with io.open(a.csv, "w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f)
            w.writerow(["group", "metric", "n", "mean", "sd", "min", "max", "seeds"])
            for r in agg_rows:
                # CSV 每行都写组名（Markdown 为了好读只在首行写），方便下游脚本解析
                w.writerow([r["group_key"], r["metric"], r["n"],
                            "%.6f" % r["mean"], "%.6f" % r["sd"],
                            "%.6f" % r["min"], "%.6f" % r["max"], r["seeds"]])
        print("written:", a.csv)


if __name__ == "__main__":
    main()
