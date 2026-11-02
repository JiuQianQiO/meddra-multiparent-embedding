# -*- coding: utf-8 -*-
r"""compile_inductive_seeds.py — 归纳式（disjoint 模型选择）多种子汇总（离线，仅标准库）。

输入：调度器 IND_* 任务的运行目录（每个目录含 holdout_metrics_<标签>.json、inductive_holdout.json、
      RUN_META.json）。这些目录由 2026-10-29 修订版 Meddra_inductive_SAGE.py / Meddra_inductive_noGNN.py
      在 MEDDRA_IND_SELECT=disjoint 下产生：模型选择集取自非留出节点，留出集只在选定权重上评估一次。

做四件事：
  1. 逐目录校验：协议必须是 disjoint；留出集代码与第一轮 IND-SAGE-KEEP 的 inductive_holdout.json
     逐成员一致（--base-holdout）；留出实例数 = 18,567；同组内训练种子不重复。
  2. 按实验分组（IND-SAGE-KEEP / IND-NOGNN-PUBMEDBERT）给出 mean ± sd（ddof=1）。
  3. 与稿件表 5 的第一轮值（在留出集上早停）对照：同为 seed 42 的差 = 选择乐观偏差的估计。
  4. 若给出 --inductive-rule（inductive_rule_baseline.py 的 JSON），给出 KEEP − I5（最近已标注祖先规则）。

用法：
  python compile_inductive_seeds.py --run-root /mnt/workspace/JBI/data/jbi_runs_review \
      --base-holdout "$IND_HOLDOUT" \
      --inductive-rule /mnt/workspace/JBI/analysis_out/INDUCTIVE_RULE_BASELINE.json \
      --md /mnt/workspace/JBI/analysis_out/INDUCTIVE_SEEDS.md \
      --json /mnt/workspace/JBI/analysis_out/INDUCTIVE_SEEDS.json
"""
import argparse
import glob
import io
import json
import os

# 稿件表 5 / 附录 B.7 的第一轮值（holdout 协议：在留出集上早停与选权重；seed 42 单次）
PAPER_LEGACY = {
    "IND-SAGE-KEEP": {"accuracy": 0.9333, "macro_f1": 0.8682, "hlt_acc": 0.9253, "pt_acc": 0.9310, "llt_acc": 0.9340},
    "IND-NOGNN-PUBMEDBERT": {"accuracy": 0.7453, "macro_f1": 0.6299, "hlt_acc": 0.8276, "pt_acc": 0.7640, "llt_acc": 0.7388},
    "IND-SAGE-MASK": {"accuracy": 0.5487, "macro_f1": 0.3580},   # 稿件表 5（附录 B.7 未列 MASK 分层值）
}
# SUBMIT 组的归纳式任务数（2026-10-29 17:20 定稿）：KEEP 5 个种子，MASK / NoGNN 各 1 个
EXPECTED_N = {"IND-SAGE-KEEP": 5, "IND-SAGE-MASK": 1, "IND-NOGNN-PUBMEDBERT": 1}
PAPER_N_INSTANCES = 18567
ACC_THRESHOLD = 0.0058     # 稿件附录 A.1 的跨配置阈值（Acc 0.58 pp）
METRICS = ("accuracy", "macro_f1", "weighted_f1", "hlt_acc", "pt_acc", "llt_acc",
           "hlt_macro_f1", "pt_macro_f1", "llt_macro_f1")
ALIASES = {"IND-PUBMEDBERT": "IND-NOGNN-PUBMEDBERT", "PUBMEDBERT": "IND-NOGNN-PUBMEDBERT"}


def _load(p):
    try:
        with io.open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _stats(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return {"n": len(vals), "mean": m, "sd": sd}


def _fmt(s, nd=4):
    if not s:
        return "—"
    return ("%.*f ± %.*f (n=%d)" % (nd, s["mean"], nd, s["sd"], s["n"]) if s["n"] >= 2
            else "%.*f (n=1)" % (nd, s["mean"]))


def _experiment(run_dir, meta):
    exp = ((meta or {}).get("env") or {}).get("MEDDRA_EXPERIMENT") or os.path.basename(run_dir)
    exp = str(exp).upper()
    return ALIASES.get(exp, exp)


def scan(run_dirs, base_codes):
    rows, problems = [], []
    for d in run_dirs:
        name = os.path.basename(os.path.normpath(d))
        hm = sorted(glob.glob(os.path.join(d, "holdout_metrics_*.json")))
        if not hm:
            if glob.glob(os.path.join(d, "best_checkpoint_*.pt")):
                problems.append("%s：有 checkpoint 但没有 holdout_metrics_*.json（多半是旧版训练器或未设 disjoint）" % name)
            else:
                problems.append("%s：尚未完成（没有 holdout_metrics_*.json）" % name)
            continue
        m = _load(hm[0]) or {}
        man = _load(os.path.join(d, "inductive_holdout.json")) or {}
        meta = _load(os.path.join(d, "RUN_META.json")) or {}
        bad = []
        if m.get("selection_protocol") != "disjoint" or man.get("selection_protocol") != "disjoint":
            bad.append("协议不是 disjoint")
        codes = man.get("holdout_node_codes")
        if base_codes is not None:
            if codes is None:
                bad.append("缺 inductive_holdout.json，无法核对留出集")
            elif set(int(c) for c in codes) != base_codes:
                bad.append("留出集与第一轮不一致（%d vs %d 个代码）" % (len(codes), len(base_codes)))
        n_inst = m.get("num_holdout_instances")
        if n_inst is not None and int(n_inst) != PAPER_N_INSTANCES:
            bad.append("留出实例数 %s ≠ 稿件 %d" % (n_inst, PAPER_N_INSTANCES))
        if bad:
            problems.append("%s：%s（已排除）" % (name, "；".join(bad)))
            continue
        row = {"run": name, "experiment": _experiment(d, meta),
               "train_seed": m.get("train_seed"), "keep_edges": m.get("keep_edges"),
               "selected_epoch": m.get("selected_epoch"),
               "select_acc": m.get("select_set_accuracy_at_selected_epoch"),
               "n_holdout": n_inst, "n_select": m.get("num_select_instances"),
               "n_train": m.get("num_train_instances")}
        for k in METRICS:
            row[k] = m.get(k)
        rows.append(row)
    return rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="", help="扫描其中 IND_* 目录")
    ap.add_argument("--runs", nargs="*", default=[], help="或直接列出运行目录")
    ap.add_argument("--base-holdout", default="", help="第一轮 IND-SAGE-KEEP 的 inductive_holdout.json")
    ap.add_argument("--inductive-rule", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    run_dirs = list(a.runs)
    if a.run_root:
        run_dirs += sorted(p for p in glob.glob(os.path.join(a.run_root, "IND_*")) if os.path.isdir(p))
    base = _load(a.base_holdout) if a.base_holdout else None
    base_codes = set(int(c) for c in base["holdout_node_codes"]) if base and "holdout_node_codes" in base else None
    rows, problems = scan(run_dirs, base_codes)
    if base_codes is None:
        problems.append("未提供 --base-holdout：没有核对留出集是否与第一轮逐成员一致")

    groups = {}
    for r in rows:
        groups.setdefault(r["experiment"], []).append(r)
    for g, rs in groups.items():
        seeds = [r["train_seed"] for r in rs]
        dup = sorted({s for s in seeds if seeds.count(s) > 1})
        if dup:
            problems.append("%s 有重复训练种子 %s（同一种子只能算一次）" % (g, dup))
        if len(set(seeds)) == 1 and len(seeds) > 1:
            problems.append("%s 的 %d 个 run 训练种子全相同——训练器没有读 MEDDRA_TRAIN_SEED" % (g, len(seeds)))

    for g, n_exp in EXPECTED_N.items():
        n_got = len(groups.get(g, []))
        if n_got < n_exp:
            problems.append("%s 只有 %d 个有效 run（SUBMIT 组应为 %d 个）" % (g, n_got, n_exp))

    stats = {g: {k: _stats([r.get(k) for r in rs]) for k in METRICS + ("select_acc",)} for g, rs in groups.items()}

    # 选择乐观偏差：同 seed 42 的 disjoint 值 vs 稿件第一轮值
    optimism = {}
    for g, ref in PAPER_LEGACY.items():
        s42 = [r for r in groups.get(g, []) if str(r["train_seed"]) == "42"]
        if s42:
            r = s42[0]
            optimism[g] = {k: (ref[k] - r[k]) if r.get(k) is not None else None for k in ref}
    gap = {}
    keep, ng, mask = stats.get("IND-SAGE-KEEP"), stats.get("IND-NOGNN-PUBMEDBERT"), stats.get("IND-SAGE-MASK")
    if keep and ng and keep["accuracy"] and ng["accuracy"]:
        gap = {"keep_minus_nognn_acc_pp": 100 * (keep["accuracy"]["mean"] - ng["accuracy"]["mean"]),
               "paper_legacy_pp": 100 * (PAPER_LEGACY["IND-SAGE-KEEP"]["accuracy"] - PAPER_LEGACY["IND-NOGNN-PUBMEDBERT"]["accuracy"])}
    if mask and ng and mask["accuracy"] and ng["accuracy"]:
        gap["mask_minus_nognn_acc_pp"] = 100 * (mask["accuracy"]["mean"] - ng["accuracy"]["mean"])
        gap["mask_paper_legacy_pp"] = 100 * (PAPER_LEGACY["IND-SAGE-MASK"]["accuracy"] - PAPER_LEGACY["IND-NOGNN-PUBMEDBERT"]["accuracy"])
    _opts = [abs(o["accuracy"]) for o in optimism.values() if o.get("accuracy") is not None]
    table5 = None
    if _opts:
        _mx = max(_opts)
        table5 = {"max_abs_optimism_pp": 100 * _mx, "switch_to_disjoint": _mx > ACC_THRESHOLD,
                  "n_rows_checked": len(_opts)}
    rule = {}
    ir = _load(a.inductive_rule) if a.inductive_rule else None
    if ir and keep and keep["accuracy"]:
        try:
            i5 = ir["metrics"]["I5"]["all"]
            rule = {"I5_acc": i5["acc"], "I5_macro_f1": i5["macro_f1"],
                    "keep_minus_i5_acc_pp": 100 * (keep["accuracy"]["mean"] - i5["acc"]),
                    "keep_minus_i5_mf1_pp": 100 * (keep["macro_f1"]["mean"] - i5["macro_f1"]) if keep["macro_f1"] else None}
        except (KeyError, TypeError):
            problems.append("--inductive-rule 文件里没有 metrics.I5.all")

    L = ["# 归纳式多种子汇总（disjoint 模型选择）", "",
         "> 模型选择集取自非留出节点；留出集（%d 个层级实例）只在选定权重上评估一次。" % PAPER_N_INSTANCES,
         "> sd 为样本标准差（ddof=1）。", ""]
    L += ["## 1. 逐 run", "", "| run | 实验 | 训练种子 | 选定轮 | 选择集 Acc | 留出 Acc | 留出 Macro-F1 | HLT / PT / LLT Acc |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda x: (x["experiment"], str(x["train_seed"]))):
        f = lambda v: "—" if v is None else "%.4f" % v
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s / %s / %s |" % (
            r["run"], r["experiment"], r["train_seed"], r["selected_epoch"], f(r["select_acc"]),
            f(r["accuracy"]), f(r["macro_f1"]), f(r["hlt_acc"]), f(r["pt_acc"]), f(r["llt_acc"])))
    L += ["", "## 2. 分组（mean ± sd）", "", "| 实验 | 留出 Acc | 留出 Macro-F1 | HLT Acc | PT Acc | LLT Acc |",
          "| --- | --- | --- | --- | --- | --- |"]
    for g, s in sorted(stats.items()):
        L.append("| %s | %s | %s | %s | %s | %s |" % (g, _fmt(s["accuracy"]), _fmt(s["macro_f1"]),
                                                   _fmt(s["hlt_acc"]), _fmt(s["pt_acc"]), _fmt(s["llt_acc"])))
    L += ["", "## 3. 选择乐观偏差（稿件第一轮值 − 同 seed 42 的 disjoint 值，pp）", ""]
    if optimism:
        L += ["| 实验 | Acc | Macro-F1 | 判定 |", "| --- | --- | --- | --- |"]
        for g, o in optimism.items():
            da = o.get("accuracy")
            verdict = ("—" if da is None else ("在 0.58 pp 阈值内" if abs(da) <= ACC_THRESHOLD else "超过 0.58 pp"))
            L.append("| %s | %s | %s | %s |" % (g, "—" if da is None else "%+.2f" % (100 * da),
                                               "—" if o.get("macro_f1") is None else "%+.2f" % (100 * o["macro_f1"]), verdict))
    else:
        L.append("- 没有 seed 42 的 disjoint 结果（SUBMIT 组的 IND_KEEP_S42 / IND_MASK_S42 / IND_NOGNN_S42 未完成）")
    if gap.get("keep_minus_nognn_acc_pp") is not None:
        L += ["", "- KEEP − NoGNN（disjoint，组均值）= %+.2f pp；稿件第一轮 = %+.2f pp" % (
            gap["keep_minus_nognn_acc_pp"], gap["paper_legacy_pp"])]
    if gap.get("mask_minus_nognn_acc_pp") is not None:
        L += ["- MASK − NoGNN（disjoint）= %+.2f pp；稿件第一轮 = %+.2f pp" % (
            gap["mask_minus_nognn_acc_pp"], gap["mask_paper_legacy_pp"])]
    if table5:
        L += ["- **表 5 口径**：三行中最大乐观偏差 %.2f pp（检查 %d 行）→ %s" % (
            table5["max_abs_optimism_pp"], table5["n_rows_checked"],
            "超过 0.58 pp：表 5 的 SAGE / NoGNN 行改用 disjoint 结果，第一轮值移到附录；GAT 行（未用新协议重跑）移出表 5 或注明第一轮协议"
            if table5["switch_to_disjoint"] else
            "在 0.58 pp 内：表 5 保留第一轮值，表注补 disjoint 复核结果")]
    if rule:
        L += ["- KEEP − I5 最近已标注祖先规则 = %+.2f pp Acc（I5 = %.4f / %.4f）" % (
            rule["keep_minus_i5_acc_pp"], rule["I5_acc"], rule["I5_macro_f1"])]
    L += ["", "## 4. 问题 / 需要注意", ""] + (["- " + p for p in problems] if problems else ["- 无"])
    md = "\n".join(L) + "\n"
    print(md)
    out = {"rows": rows, "stats": stats, "optimism_vs_paper": optimism, "keep_vs_nognn": gap, "table5": table5,
           "keep_vs_i5": rule, "problems": problems}
    if a.md:
        io.open(a.md, "w", encoding="utf-8", newline="\n").write(md)
        print("written:", a.md)
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="\n").write(json.dumps(out, ensure_ascii=False, indent=1))
        print("written:", a.json)
    return out


if __name__ == "__main__":
    main()
