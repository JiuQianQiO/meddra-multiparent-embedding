# -*- coding: utf-8 -*-
r"""make_paper_numbers.py — 汇总补充实验与离线分析的全部产物，生成正文修订稿需要的数字（离线，无 GPU）。

对应 `paper/JBI/JBI_正文修订_E0_kNN_多轴_2026-10-29.md` 中的 [[占位符]]，并给出：
  1. 占位符 → 数值 → 来源 的对照表；
  2. E0 结论句用情况 A 还是情况 B 的判定（及理由）；
  3. SOC 控制表（E1A / E1B / SOCDROP vs V1，分层 Acc 与 Macro-F1，mean ± sd）；
  4. 多种子表（V1 / V2 / C7 / C3 / V3 / V4）与 0.93 pp 阈值、V1 ±2 sd 的比较；
  5. 缺失输入清单（哪一步还没跑）。

所有输入都是可选的：缺哪个就在"缺失输入"里列出来，其余照常输出。

用法：
  python make_paper_numbers.py \
    --review-root /mnt/workspace/JBI/data/jbi_runs_review \
    --base-root   /mnt/workspace/JBI/data/jbi_runs \
    --rule-neighbor /mnt/workspace/JBI/analysis_out/RULE_BASELINE_NEIGHBOR.json \
    --b1-dir      baseline/baseline_outputs \
    --multiaxial  /mnt/workspace/JBI/analysis_out/MULTIAXIAL_SUMMARY.json \
    --inductive-rule /mnt/workspace/JBI/analysis_out/INDUCTIVE_RULE_BASELINE.json \
    --md /mnt/workspace/JBI/analysis_out/PAPER_NUMBERS.md --json /mnt/workspace/JBI/analysis_out/PAPER_NUMBERS.json
"""
import argparse
import io
import json
import os

from compile_review_seed_metrics import scan, scan_base, check_duplicate_seeds

THRESH = 0.0093          # 稿件附录 A.1 的 0.93 pp 重采样阈值（Macro-F1）
# 判定规则：情况 A = V1（n=5 均值）比最强规则 R6 高 ≥ 1 pp，且差值大于 V1 的 2 sd；否则情况 B。
V1_PAPER = {"acc": 0.9932, "macro_f1": 0.9892}   # 稿件主结果（seed 42，单次）
KNN_PAPER = {"bge": 0.9372, "e5": 0.9244}         # 稿件表 2（kNN 只有 accuracy）
METRICS = ("acc", "macro_f1", "hlgt", "hlt", "pt", "llt", "hlgt_acc", "hlt_acc", "pt_acc", "llt_acc")


def _load(path):
    if not path or not os.path.isfile(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return {"n": len(vals), "mean": m, "sd": sd}


def _f(s, nd=4):
    if not s:
        return "—"
    if s["n"] >= 2:
        return "%.*f ± %.*f (n=%d)" % (nd, s["mean"], nd, s["sd"], s["n"])
    return "%.*f (n=1)" % (nd, s["mean"])


def _pp(x):
    return "—" if x is None else "%+.2f pp" % (100 * x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-root", default="")
    ap.add_argument("--base-root", default="")
    ap.add_argument("--rule-neighbor", default="")
    ap.add_argument("--b1-dir", default="")
    ap.add_argument("--multiaxial", default="")
    ap.add_argument("--inductive-rule", default="")
    ap.add_argument("--inductive-seeds", default="", help="compile_inductive_seeds.py 的 JSON（2026-10-29 第二轮）")
    ap.add_argument("--joint-agg", default="", help="aggregate_joint_diag.py 的 JSON（2026-10-29 第二轮）")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    missing, ph = [], {}

    def put(key, value, source):
        ph[key] = {"value": value, "source": source}

    # ---------------- 多种子 ----------------
    rows = []
    if a.review_root and os.path.isdir(a.review_root):
        rows += scan(a.review_root)
    else:
        missing.append("--review-root（补充实验结果根目录）")
    if a.base_root and os.path.isdir(a.base_root):
        rows += scan_base(a.base_root)
    else:
        missing.append("--base-root（第一轮主结果根目录，提供 seed 42）")
    if rows:
        dups = check_duplicate_seeds(rows)
        if dups:
            missing.append("有重复种子（见上方 [warn]），多种子统计不可信，先处理")
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    G = {g: {m: _stats([r.get(m) for r in rs]) for m in METRICS} for g, rs in groups.items()}

    def gm(g, m):
        return (G.get(g) or {}).get(m)

    v1_acc = gm("V1", "acc")
    v1_mf1 = gm("V1", "macro_f1")
    v1_acc_ref = v1_acc["mean"] if v1_acc else V1_PAPER["acc"]
    v1_mf1_ref = v1_mf1["mean"] if v1_mf1 else V1_PAPER["macro_f1"]
    if not v1_acc or v1_acc["n"] < 5:
        missing.append("V1 不足 5 个种子（当前 n=%s），G1 的 V1_S43–46 未跑完或基准目录缺失" % (v1_acc["n"] if v1_acc else 0))

    # ---------------- E0b 邻居标签规则 ----------------
    rn = _load(a.rule_neighbor)
    r6_all = r6_multi = None
    if rn:
        met = rn.get("metrics", {})
        for rid in ("R5", "R6"):
            for g, key in (("all", ""), ("single-axis (n_SOC=1)", " single-axis"), ("multiaxial (n_SOC>=2)", " multiaxial")):
                v = met.get(rid, {}).get(g)
                if v:
                    put("%s%s Acc/MF1" % (rid, key), "%.4f/%.4f" % (v["acc"], v["macro_f1"]), "RULE_BASELINE_NEIGHBOR.json")
        put("R6 coverage", "%.2f" % (100 * rn.get("coverage", {}).get("R6", 0)), "RULE_BASELINE_NEIGHBOR.json")
        r6_all = met.get("R6", {}).get("all")
        r6_multi = met.get("R6", {}).get("multiaxial (n_SOC>=2)")
        best_rule = max((met[r]["all"]["acc"], r) for r in ("R2", "R3", "R4", "R5", "R6") if r in met and "all" in met[r])
        put("strongest rule (no test-node label fields)", "%s %.4f" % (best_rule[1], best_rule[0]), "RULE_BASELINE_NEIGHBOR.json")
        # kNN 近邻关系
        for kp, kres in (rn.get("knn") or {}).items():
            tag = "BGE" if "bge" in kp.lower() else ("E5" if "e5" in kp.lower() else os.path.basename(kp))
            rel = kres.get("by_relation", {})
            hier = {k: rel[k] for k in ("parent", "child", "sibling") if k in rel}
            n_h = sum(v["n"] for v in hier.values())
            n_all = kres.get("n") or 1
            put("kNN-%s share_hier" % tag, "%.2f" % (100.0 * n_h / n_all), "RULE_BASELINE_NEIGHBOR.json --knn-pred")
            for k in ("child", "sibling", "parent"):
                if k in rel:
                    put("kNN-%s share_%s" % (tag, "sib" if k == "sibling" else k), "%.2f" % (100 * rel[k]["share"]), "同上")
            if n_h:
                put("kNN-%s acc_hier" % tag, "%.4f" % (sum(v["n"] * v["acc"] for v in hier.values()) / n_h), "同上")
            if "other" in rel:
                put("kNN-%s acc_other" % tag, "%.4f" % rel["other"]["acc"], "同上")
            ax = kres.get("by_axis", {})
            for name, key in (("single-axis (n_SOC=1)", "single-axis"), ("multiaxial (n_SOC>=2)", "multiaxial")):
                if name in ax:
                    put("kNN-%s %s" % (tag, key), "%.4f" % ax[name]["acc"], "同上")
    else:
        missing.append("--rule-neighbor（rule_baseline_neighbor.py 的 JSON 输出）")

    # ---------------- B1 kNN ----------------
    knn = {}
    if a.b1_dir and os.path.isdir(a.b1_dir):
        for m in ("bge", "e5"):
            j = _load(os.path.join(a.b1_dir, "result_B1_%s.json" % m))
            if not j:
                missing.append("%s/result_B1_%s.json" % (a.b1_dir, m))
                continue
            k = (j.get("extra") or {}).get("knn") or {}
            knn[m] = k
            if k.get("macro_f1") is None:
                missing.append("result_B1_%s.json 的 kNN 没有 macro_f1（还是补丁前的旧结果，需重跑 train_b1_text_embed.py）" % m)
            if k.get("accuracy") is not None and abs(k["accuracy"] - KNN_PAPER[m]) > 5e-4:
                missing.append("kNN-%s accuracy=%.4f 与稿件 %.4f 不一致，先核对再写入" % (m, k["accuracy"], KNN_PAPER[m]))
            put("kNN-%s Acc/MF1" % m.upper(), "%s/%s" % (
                "%.4f" % k["accuracy"] if k.get("accuracy") is not None else "—",
                "%.4f" % k["macro_f1"] if k.get("macro_f1") is not None else "—"), "result_B1_%s.json" % m)
        if knn.get("bge", {}).get("macro_f1") is not None:
            put("ΔMF1 vs kNN-BGE (paper V1)", "%.2f" % (100 * (V1_PAPER["macro_f1"] - knn["bge"]["macro_f1"])),
                "0.9892 − kNN-BGE Macro-F1")
            if v1_mf1 and v1_mf1["n"] >= 2:
                put("ΔMF1 vs kNN-BGE (V1 mean)", "%.2f" % (100 * (v1_mf1_ref - knn["bge"]["macro_f1"])), "V1 n=%d 均值" % v1_mf1["n"])
            put("Δacc vs kNN-BGE (paper V1)", "%.2f" % (100 * (V1_PAPER["acc"] - knn["bge"]["accuracy"])), "0.9932 − kNN-BGE Acc")
    else:
        missing.append("--b1-dir（baseline/baseline_outputs）")

    # ---------------- 多轴 ----------------
    mx = _load(a.multiaxial)
    v1_multi = None
    if mx:
        for gname, ginfo in (mx.get("groups") or {}).items():
            ag = ginfo.get("agg")
            if not ag:
                continue
            put("%s single-axis Acc" % gname, _f(ag.get("single_acc")), "MULTIAXIAL_SUMMARY.json")
            put("%s multiaxial Acc" % gname, _f(ag.get("multi_acc")), "同上")
            put("%s multiaxial MF1" % gname, _f(ag.get("multi_macro_f1")), "同上")
            put("%s err_share_multi" % gname, _f(ag.get("err_share_multi")), "同上")
            if gname == "V1":
                v1_multi = ag.get("multi_acc")
                if ag.get("n_err"):
                    put("n_err (V1)", "%.1f" % ag["n_err"]["mean"], "同上（各种子均值）")
                    put("n_err_multi (V1)", "%.1f" % ag["n_err_multi"]["mean"], "同上（各种子均值）")
        if "V1" not in (mx.get("groups") or {}):
            missing.append("MULTIAXIAL_SUMMARY.json 中没有 V1 组")
    else:
        missing.append("--multiaxial（multiaxial_summary.py 的 JSON 输出）")

    # ---------------- 归纳式规则 ----------------
    ir = _load(a.inductive_rule)
    if ir:
        i5 = ir["metrics"]["I5"]["all"]
        put("inductive ancestor-label rule (I5) Acc/MF1", "%.4f/%.4f" % (i5["acc"], i5["macro_f1"]), "INDUCTIVE_RULE_BASELINE.json")
        put("IND-SAGE-KEEP − I5 (Acc)", "%.2f" % (100 * ir["gap_ind_sage_keep_minus_i5_acc"]), "同上（模型值取自稿件表 4）")
        for w in ir.get("warnings", []):
            missing.append("归纳式规则：" + w)
    else:
        missing.append("--inductive-rule（inductive_rule_baseline.py 的 JSON 输出；只在情况 B 必需）")

    # ---------------- 情况 A / B 判定 ----------------
    decision = {"case": None, "reason": ""}
    if r6_all:
        d = v1_acc_ref - r6_all["acc"]
        sd = v1_acc["sd"] if (v1_acc and v1_acc["n"] >= 2) else None
        put("Δ overall (V1 − R6, Acc)", "%.2f" % (100 * d), "V1 %s − R6" % ("n=%d 均值" % v1_acc["n"] if v1_acc else "稿件值"))
        if r6_multi and v1_multi:
            put("Δ multiaxial (V1 − R6, Acc)", "%.2f" % (100 * (v1_multi["mean"] - r6_multi["acc"])), "MULTIAXIAL_SUMMARY − R6 multiaxial")
        clear = d >= 0.01 and (sd is None or d > 2 * sd)
        if clear:
            decision = {"case": "A", "reason": "V1 比最强规则 R6 高 %.2f pp（≥ 1 pp%s）" % (
                100 * d, "，且大于 V1 的 2 sd = %.2f pp" % (200 * sd) if sd is not None else "；V1 尚无多种子 sd")}
        else:
            decision = {"case": "B", "reason": "V1 − R6 = %+.2f pp（未达到 ≥ 1 pp 且超出 V1 ±2 sd 的标准%s）" % (
                100 * d, "，2 sd = %.2f pp" % (200 * sd) if sd is not None else "")}
        if sd is None:
            decision["reason"] += "；V1 只有单次观察，判定是暂定的，G1 跑完后重算"
    else:
        decision["reason"] = "缺少 R6 结果，无法判定"

    # ---------------- 输出 ----------------
    L = ["# 正文修订稿所需数字（自动汇总）", ""]
    L += ["## 1. E0 结论句：情况 %s" % (decision["case"] or "未定"), "", "- " + decision["reason"], ""]

    L += ["## 2. 占位符", "", "| 占位符 | 值 | 来源 |", "| --- | --- | --- |"]
    for k, v in ph.items():
        L.append("| %s | %s | %s |" % (k, v["value"], v["source"]))
    L.append("")

    L += ["## 3. SOC 控制（分层，mean ± sd；Δ 相对 V1 均值）", "",
          "> 3 层 GNN 下 LLT 看不到 SOC 节点，以 PT（n=2,738）为主要读数。`!` 表示 |Δ| 超过 V1 的 2 sd。", "",
          "| 组 | 指标 | HLGT | HLT | PT | LLT | 总体 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    soc_tab = {}
    for g in ("V1", "E1A", "E1B", "SOCDROP"):
        if g not in G:
            continue
        for kind, keys in (("Acc", ("hlgt_acc", "hlt_acc", "pt_acc", "llt_acc", "acc")),
                           ("Macro-F1", ("hlgt", "hlt", "pt", "llt", "macro_f1"))):
            cells = []
            for k in keys:
                s = gm(g, k)
                base = gm("V1", k)
                if g == "V1" or not s or not base:
                    cells.append(_f(s))
                    continue
                d = s["mean"] - base["mean"]
                flag = "!" if base["n"] >= 2 and abs(d) > max(2 * base["sd"], 0.0005) else ""
                cells.append("%s（%s%s）" % (_f(s), _pp(d), flag))
                soc_tab.setdefault(g, {})[k] = {"delta": d, "flag": bool(flag)}
            L.append("| %s | %s | %s |" % (g, kind, " | ".join(cells)))
    if not any(g in G for g in ("E1A", "E1B", "SOCDROP")):
        missing.append("没有 SOC 控制结果（E1B / SOCDROP 属于 G1）")
    L.append("")

    L += ["## 4. 多种子主表（总体 Acc / Macro-F1）", "",
          "| 组 | Acc | Macro-F1 | ΔMF1 vs V1 | 与阈值 / V1 ±2 sd 的比较 |", "| --- | --- | --- | --- | --- |"]
    seed_tab = {}
    for g in ("V1", "V2", "C7", "C3", "V3", "V4", "E1A", "E1B", "SOCDROP", "NGPB", "GAT"):
        if g not in G:
            continue
        a_, f_ = gm(g, "acc"), gm(g, "macro_f1")
        if g == "V1" or not f_ or not v1_mf1:
            L.append("| %s | %s | %s | — | — |" % (g, _f(a_), _f(f_)))
            continue
        d = f_["mean"] - v1_mf1["mean"]
        notes = ["超过 0.93 pp 阈值" if abs(d) > THRESH else "在 0.93 pp 阈值内"]
        if v1_mf1["n"] >= 2:
            notes.append("超出 V1 ±2 sd" if abs(d) > max(2 * v1_mf1["sd"], 0.0005) else "在 V1 ±2 sd 内")
        seed_tab[g] = {"delta_mf1": d, "notes": notes}
        L.append("| %s | %s | %s | %s | %s |" % (g, _f(a_), _f(f_), _pp(d), "；".join(notes)))
    L.append("")


    # ---------------- 稿件占位符（final_submission 中的 [[...]]，2026-10-29 新增） ----------------
    # 键名与 final_submission/main/JBI_Manuscript_EN_submission.(md|docx) 和
    # JBI_Supplementary_EN.(md|docx) 中黄色高亮的 [[...]] 完全一致，可逐个查找替换。
    def _v(key, part=None):
        v = ph.get(key, {}).get("value")
        if v is None:
            return None
        return v.split("/")[part] if part is not None and "/" in v else v

    def _ms(g, m):
        st = gm(g, m)
        return None if not st else _f(st)

    def _soc_d(g, k):
        d = (soc_tab.get(g) or {}).get(k)
        return None if not d else "%+.2f%s" % (100 * d["delta"], " (beyond V1 ±2 sd)" if d["flag"] else "")

    def _knn(m, f):
        x = (knn.get(m) or {}).get(f)
        return None if x is None else "%.4f" % x

    MS = [
        ("ΔMF1 vs kNN-BGE", _v("ΔMF1 vs kNN-BGE (paper V1)"), "0.9892 − kNN-BGE Macro-F1（摘要、4.1、B.3）"),
        ("kNN-BGE MF1", _knn("bge", "macro_f1"), "result_B1_bge.json"),
        ("kNN-BGE WF1", _knn("bge", "weighted_f1"), "result_B1_bge.json"),
        ("kNN-E5 MF1", _knn("e5", "macro_f1"), "result_B1_e5.json"),
        ("kNN-E5 WF1", _knn("e5", "weighted_f1"), "result_B1_e5.json"),
        ("R5 Acc/MF1", _v("R5 Acc/MF1"), "RULE_BASELINE_NEIGHBOR.json"),
        ("R6 Acc/MF1", _v("R6 Acc/MF1"), "同上"),
        ("R6 coverage", _v("R6 coverage"), "同上"),
        ("R5 single-axis Acc", _v("R5 single-axis Acc/MF1", 0), "同上"),
        ("R5 multiaxial Acc", _v("R5 multiaxial Acc/MF1", 0), "同上"),
        ("R6 single-axis Acc", _v("R6 single-axis Acc/MF1", 0), "同上"),
        ("R6 multiaxial Acc", _v("R6 multiaxial Acc/MF1", 0), "同上"),
        ("kNN-BGE share_hier", _v("kNN-BGE share_hier"), "RULE_BASELINE_NEIGHBOR.json --knn-pred"),
        ("kNN-BGE share_child", _v("kNN-BGE share_child"), "同上"),
        ("kNN-BGE share_sib", _v("kNN-BGE share_sib"), "同上"),
        ("kNN-BGE share_parent", _v("kNN-BGE share_parent"), "同上"),
        ("kNN-BGE acc_hier", _v("kNN-BGE acc_hier"), "同上"),
        ("kNN-BGE acc_other", _v("kNN-BGE acc_other"), "同上"),
        ("kNN-BGE single-axis", _v("kNN-BGE single-axis"), "同上"),
        ("kNN-BGE multiaxial", _v("kNN-BGE multiaxial"), "同上"),
        ("V1 Acc/MF1", (lambda a_, f_: None if not (a_ and f_) else "%s / %s" % (a_, f_))(_ms("V1", "acc"), _ms("V1", "macro_f1")), "多种子（n=5）"),
        ("V2 Acc/MF1", (lambda a_, f_: None if not (a_ and f_) else "%s / %s" % (a_, f_))(_ms("V2", "acc"), _ms("V2", "macro_f1")), "多种子"),
        ("C7 Acc/MF1", (lambda a_, f_: None if not (a_ and f_) else "%s / %s" % (a_, f_))(_ms("C7", "acc"), _ms("C7", "macro_f1")), "多种子"),
        ("V1 single-axis Acc", _v("V1 single-axis Acc"), "MULTIAXIAL_SUMMARY.json"),
        ("V1 multiaxial Acc", _v("V1 multiaxial Acc"), "同上"),
        ("V1 multiaxial MF1", _v("V1 multiaxial MF1"), "同上"),
        ("E1B ΔPT MF1", _soc_d("E1B", "pt"), "§3 SOC 控制，PT Macro-F1，pp"),
        ("SOCDROP ΔPT MF1", _soc_d("SOCDROP", "pt"), "同上"),
        ("Δ overall", _v("Δ overall (V1 − R6, Acc)"), "V1 均值 − R6"),
        ("Δ multiaxial", _v("Δ multiaxial (V1 − R6, Acc)"), "V1 多轴均值 − R6 多轴"),
        ("I5 Acc/MF1", _v("inductive ancestor-label rule (I5) Acc/MF1"), "INDUCTIVE_RULE_BASELINE.json（仅情况 B 需要）"),
    ]

    # 附录 A.4 表 A1（多种子 + SOC 控制）。C3 在 SKIP 组（不跑）：稿件表 A1 删除 C3 行，不计入缺失
    for g in ("V1", "V2", "C7", "E1B", "SOCDROP", "C3", "V3", "V4", "E1A"):
        opt = g == "C3"
        _src = "附录表 A1" + ("（SKIP 组，不跑：稿件删除 C3 行，可选）" if opt else "")
        for lab, m in (("Acc", "acc"), ("MF1", "macro_f1"), ("HLGT F1", "hlgt"), ("HLT F1", "hlt"), ("PT F1", "pt")):
            MS.append(("%s %s" % (g, lab), _ms(g, m), _src))
        st = seed_tab.get(g)
        MS.append(("%s ΔMF1" % g, "—" if g == "V1" else (None if not st else "%+.2f（%s）" % (100 * st["delta_mf1"], "；".join(st["notes"]))),
                   _src))
    # kNN 近邻关系：某类关系不存在时占比记 0.00
    if ph.get("kNN-BGE share_hier"):
        MS = [(k, ("0.00" if (v is None and k.startswith("kNN-BGE share_")) else v), src) for k, v, src in MS]
    L += ["## 4b. final_submission 稿件占位符（键名与稿件中黄色 [[...]] 一致）", "",
          "> 稿件里另有两处条件块：`[[CASE A — …]]` / `[[CASE B — …]]` 按 §1 保留其一并删掉方括号；"
          "`[[If share_hier < 50%: …]]`（附录 B.3）按 kNN-BGE share_hier 处理后删除。", "",
          "| 稿件占位符 | 值 | 来源 |", "| --- | --- | --- |"]
    ms_out = {}
    for k, v, src in MS:
        ms_out[k] = v
        L.append("| `[[%s]]` | %s | %s |" % (k, v if v is not None else "**缺**", src))
        if v is None and not (k == "I5 Acc/MF1" and decision.get("case") == "A") and "可选" not in src:
            missing.append("稿件占位符 [[%s]] 无值（来源：%s）" % (k, src))
    L.append("")


    # ---------------- 4c. 归纳式 disjoint 与联合诊断多种子（2026-10-29 第二轮） ----------------
    extra = {}
    L += ["## 4c. 归纳式（disjoint 模型选择）与父×子联合诊断（多种子）", ""]
    isd = _load(a.inductive_seeds)
    ind_opt = None
    if isd:
        _ks = (isd.get("stats") or {}).get("IND-SAGE-KEEP") or {}
        _ns = (isd.get("stats") or {}).get("IND-NOGNN-PUBMEDBERT") or {}
        for nm_, st_ in (("IND-SAGE-KEEP", _ks), ("IND-NOGNN-PUBMEDBERT", _ns)):
            if st_.get("accuracy"):
                extra["%s disjoint Acc" % nm_] = _f(st_["accuracy"])
                extra["%s disjoint MF1" % nm_] = _f(st_.get("macro_f1"))
        _op = (isd.get("optimism_vs_paper") or {}).get("IND-SAGE-KEEP") or {}
        if _op.get("accuracy") is not None:
            ind_opt = _op["accuracy"]
            extra["IND-SAGE-KEEP 选择乐观偏差 (pp, 第一轮 − disjoint, seed 42)"] = "%+.2f" % (100 * ind_opt)
        for _nm in ("IND-SAGE-MASK", "IND-NOGNN-PUBMEDBERT"):
            _o2 = (isd.get("optimism_vs_paper") or {}).get(_nm) or {}
            if _o2.get("accuracy") is not None:
                extra["%s 选择乐观偏差 (pp, seed 42)" % _nm] = "%+.2f" % (100 * _o2["accuracy"])
        _kg = isd.get("keep_vs_nognn") or {}
        if _kg.get("keep_minus_nognn_acc_pp") is not None:
            extra["KEEP − NoGNN (disjoint, pp)"] = "%+.2f" % _kg["keep_minus_nognn_acc_pp"]
        if _kg.get("mask_minus_nognn_acc_pp") is not None:
            extra["MASK − NoGNN (disjoint, pp)"] = "%+.2f" % _kg["mask_minus_nognn_acc_pp"]
        if isd.get("keep_vs_i5"):
            extra["KEEP − I5 (disjoint, pp)"] = "%+.2f" % isd["keep_vs_i5"]["keep_minus_i5_acc_pp"]
        for p_ in isd.get("problems", []):
            missing.append("归纳式多种子：" + p_)
    else:
        missing.append("--inductive-seeds（compile_inductive_seeds.py 的 JSON；SUBMIT 组的 IND_* 任务）")
    jag = _load(a.joint_agg)
    if jag:
        _jv = ((jag.get("groups") or {}).get("V1") or {}).get("stats") or {}
        for k_, lab_ in (("joint_pp", "V1 联合差 (pp)"), ("parent_comp_pp", "V1 父链分量 (pp)"),
                         ("child_comp_pp", "V1 子链分量 (pp)"), ("parent_net_pp", "V1 一维父链净效应 (pp)")):
            s_ = _jv.get(k_)
            if s_:
                extra[lab_] = ("%+.2f ± %.2f (n=%d)" % (s_["mean"], s_["sd"], s_["n"]) if s_["n"] >= 2
                               else "%+.2f (n=1)" % s_["mean"])
        for p_ in jag.get("problems", []):
            missing.append("联合诊断：" + p_)
    else:
        missing.append("--joint-agg（aggregate_joint_diag.py 的 JSON）")

    # ---- V4 − V3：随机特征下对比学习的补偿作用（2026-10-29：用户决定保留该句，G2 必跑） ----
    _v3a, _v4a, _v3f, _v4f = gm("V3", "acc"), gm("V4", "acc"), gm("V3", "macro_f1"), gm("V4", "macro_f1")
    v34 = {}
    if _v3f and _v4f:
        _d = _v4f["mean"] - _v3f["mean"]
        _pool = ((_v3f["sd"] ** 2 + _v4f["sd"] ** 2) ** 0.5) if (_v3f["n"] >= 2 and _v4f["n"] >= 2) else None
        _ok = abs(_d) > THRESH and (_pool is None or abs(_d) > 2 * _pool)
        v34 = {"delta_mf1": _d, "pooled_sd": _pool, "n_v3": _v3f["n"], "n_v4": _v4f["n"], "supported": bool(_ok and _d > 0)}
        extra["V4 − V3 Macro-F1 (pp)"] = "%+.2f（V3 n=%d，V4 n=%d%s）" % (
            100 * _d, _v3f["n"], _v4f["n"], "，合并 sd = %.2f pp" % (100 * _pool) if _pool is not None else "")
        if _v3a and _v4a:
            extra["V4 − V3 Acc (pp)"] = "%+.2f" % (100 * (_v4a["mean"] - _v3a["mean"]))
        extra["'补偿作用'一句"] = ("可保留：V4 − V3 > 0.93 pp 且大于 2 倍合并 sd" if v34["supported"] else
                                 "不成立：V4 − V3 未同时超过 0.93 pp 与 2 倍合并 sd，稿件需改为'未观察到稳定的补偿作用'")
        if _v3f["n"] < 5 or _v4f["n"] < 5:
            missing.append("V3 / V4 不足 5 个种子（V3 n=%d，V4 n=%d）：SUBMIT 组的 V3_S43–46 / V4_S43–46 未跑完" % (_v3f["n"], _v4f["n"]))
    else:
        missing.append("没有 V3 / V4 多种子结果（SUBMIT 组：V3_S43–46、V4_S43–46）")
    if extra:
        L += ["| 数字 | 值 |", "| --- | --- |"] + ["| %s | %s |" % (k_, v_) for k_, v_ in extra.items()]
    else:
        L.append("- 暂无")
    L.append("")
    L += ["## 5. 缺失输入 / 需要注意", ""]
    L += ["- " + m for m in missing] if missing else ["- 无"]
    L.append("")

    # ---------------- 6. SUBMIT 组完整性与结论判定（2026-10-29 17:20：所有需要的实验一次跑完，不再有条件组） ----------------
    _need = {"V1": 5, "V2": 5, "C7": 5, "V3": 5, "V4": 5,
             "E1B": 3, "SOCDROP": 3, "E1A": 3, "NGPB": 3, "GAT": 3}
    for g_, n_ in _need.items():
        st_ = gm(g_, "macro_f1")
        if not st_ or st_["n"] < n_:
            missing.append("SUBMIT 组未完成：%s 只有 n=%d（应为 %d）" % (g_, st_["n"] if st_ else 0, n_))
    concl = {}
    # (1) 近循环：E1B / SOCDROP / E1A 的 HLT、PT 分层变化
    def _drops(g_):
        out_ = []
        for k_ in ("pt_acc", "pt", "hlt_acc", "hlt"):
            d_ = (soc_tab.get(g_) or {}).get(k_)
            if d_ and d_["flag"] and d_["delta"] < 0:
                out_.append("%s %+.2f pp" % (k_, 100 * d_["delta"]))
        return out_
    if not any(g in G for g in ("E1B", "SOCDROP", "E1A")):
        concl["近循环（SOC 控制）"] = "还没有 E1B / SOCDROP / E1A 结果"
    else:
        _z, _dp, _pm = _drops("E1B"), _drops("SOCDROP"), _drops("E1A")
        if not (_z or _dp or _pm):
            concl["近循环（SOC 控制）"] = ("三种控制在 HLT / PT 上都没有超出 V1 ±2 sd 的下降：SOC 名称、身份与结构都不是分类捷径；"
                                       "E1A 的近零效应与设计预期一致")
        else:
            _parts = []
            if _z:
                _parts.append("名称清零 E1B 下降（%s）" % "；".join(_z))
            if _dp:
                _parts.append("删除 SOC 节点 SOCDROP 下降（%s）" % "；".join(_dp))
            if _pm:
                _parts.append("名称置换 E1A 也下降（%s）→ 名称的语义内容在起作用" % "；".join(_pm))
            elif _z:
                _parts.append("E1A 无明显下降 → 起作用的是 SOC 节点的区分性输入（身份），而不是名称语义")
            concl["近循环（SOC 控制）"] = "；".join(_parts) + "。稿件 5.5(4) 需如实写明该依赖，并按 PT 层报告"
    # (2) E0 情况 A / B
    concl["E0 结论句"] = "情况 %s：%s" % (decision.get("case") or "未定", decision.get("reason") or "缺 R6 或 V1 多种子")
    # (3) 表 5 口径（归纳式选择乐观偏差）
    _t5 = (isd or {}).get("table5") if isd else None
    if _t5:
        concl["表 5 口径"] = ("最大乐观偏差 %.2f pp > 0.58 pp：表 5 的 SAGE / NoGNN 行改用 disjoint 结果，第一轮值移到附录；"
                            "GAT 归纳式两行未用新协议重跑，移出表 5 或注明第一轮协议" % _t5["max_abs_optimism_pp"]
                            if _t5["switch_to_disjoint"] else
                            "最大乐观偏差 %.2f pp ≤ 0.58 pp：表 5 保留第一轮值，表注补 disjoint 复核" % _t5["max_abs_optimism_pp"])
    else:
        concl["表 5 口径"] = "还没有归纳式 disjoint 结果"
    # (4) 情况 B 时的泛化证据：KEEP（n=5）对最近已标注祖先规则 I5
    if isd and isd.get("keep_vs_i5"):
        _ks = ((isd.get("stats") or {}).get("IND-SAGE-KEEP") or {}).get("accuracy") or {}
        _gap = isd["keep_vs_i5"]["keep_minus_i5_acc_pp"]
        _sd = 100 * _ks.get("sd", 0.0) if _ks.get("n", 0) >= 2 else None
        _clear = _gap >= 1.0 and (_sd is None or _gap > 2 * _sd)
        concl["归纳式泛化证据（KEEP − I5）"] = "%+.2f pp（KEEP n=%d%s）→ %s" % (
            _gap, _ks.get("n", 0), "，2 sd = %.2f pp" % (2 * _sd) if _sd is not None else "",
            "成立：情况 B 的泛化论证可以写" if _clear else "不明显：情况 B 时稿件不能声称归纳式优于规则，只能写'与规则相当'")
    # (5) 随机特征下对比学习的补偿作用
    concl["'补偿作用'一句（4.2 / 5.1）"] = (("保留：V4 − V3 = %+.2f pp" % (100 * v34["delta_mf1"])) if v34.get("supported")
                                     else ("改写为'未观察到稳定的补偿作用'（V4 − V3 = %+.2f pp）" % (100 * v34["delta_mf1"]) if v34
                                           else "还没有 V3 / V4 多种子结果"))
    L += ["", "## 6. SUBMIT 组完整性与结论判定", "",
          "> 2026-10-29 17:20 起只有 SUBMIT / SKIP 两组，SUBMIT 一次跑完，这里不再判定\"要不要再跑\"，只给出各结论句的写法。", "",
          "| 事项 | 判定 |", "| --- | --- |"]
    for k_, v_ in concl.items():
        L.append("| %s | %s |" % (k_, v_))
    L.append("")
    md = "\n".join(L)
    print(md)
    out = {"decision": decision, "placeholders": ph, "groups": G, "soc_control": soc_tab,
           "seed_table": seed_tab, "manuscript_placeholders": ms_out, "extra_numbers": extra, "v4_minus_v3": v34,
           "conclusions": concl, "missing": missing}
    if a.md:
        io.open(a.md, "w", encoding="utf-8", newline="\n").write(md)
        print("written:", a.md)
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="\n").write(json.dumps(out, ensure_ascii=False, indent=1))
        print("written:", a.json)
    return out


if __name__ == "__main__":
    main()
