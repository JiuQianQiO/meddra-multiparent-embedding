# -*- coding: utf-8 -*-
r"""aggregate_joint_diag.py — 父×子二维联合诊断的多种子汇总（离线，仅标准库）。

输入：diagnose_joint_subgroup.py 写出的 joint_subgroup_diag*.json（finalize_review_analysis.py 的
      --joint-dirs 产物，位于 --out-dir，文件名 joint_subgroup_diag_<运行目录名>.json）。

对每个文件取「仅有子节点的节点」口径（稿件表 4 / 附录表 B3 的口径），四格记为
  A = 父∈子∈，B = 父∈子∉，C = 父∉子∈，D = 父∉子∉；
  联合差 = A − D，分解为 父链分量 A − C 与 子链分量 C − D（A − D = (A − C) + (C − D)），
另取一维父链净效应（父∈train − 父∉train，全诊断集）与 PT 层的同一差值。
复现校验未通过（reproduce.reproduced=False）的文件一律排除。

用法：
  python aggregate_joint_diag.py \
      --group V1 /mnt/workspace/JBI/analysis_out/joint_subgroup_diag_V1_S4*.json \
                 /mnt/workspace/JBI/analysis_out/joint_subgroup_diag_C1V1.json \
      --md /mnt/workspace/JBI/analysis_out/JOINT_DIAG_SEEDS.md \
      --json /mnt/workspace/JBI/analysis_out/JOINT_DIAG_SEEDS.json
"""
import argparse
import io
import json
import os

# 稿件表 4（seed 42 单次）
PAPER_V1 = {"joint_pp": 2.23, "parent_comp_pp": -0.08, "child_comp_pp": 2.31, "parent_net_pp": 1.44}
KEYS = ("joint_pp", "parent_comp_pp", "child_comp_pp", "parent_net_pp", "parent_net_pt_pp")


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


def _fmt(s):
    if not s:
        return "—"
    return "%+.2f ± %.2f (n=%d)" % (s["mean"], s["sd"], s["n"]) if s["n"] >= 2 else "%+.2f (n=1)" % s["mean"]


def extract(path):
    d = _load(path)
    if not d:
        return None, "无法读取"
    if not (d.get("reproduce") or {}).get("reproduced"):
        return None, "复现校验未通过"
    js = d.get("joint_subgroup") or {}
    cells = (js.get("joint_has_child_only") or {}).get("cells") or {}
    acc = {k: (cells.get(k) or {}).get("acc") for k in ("父∈子∈", "父∈子∉", "父∉子∈", "父∉子∉")}
    if any(v is None for v in (acc["父∈子∈"], acc["父∉子∈"], acc["父∉子∉"])):
        return None, "四格缺值"
    pc = js.get("parent_cells") or {}
    pbl = js.get("parent_by_level") or {}
    pt = pbl.get("pt") or pbl.get("PT") or {}
    row = {
        "file": os.path.basename(path), "run_dir": d.get("run_dir"), "train_seed": d.get("train_seed"),
        "n_cells": {k: (cells.get(k) or {}).get("n") for k in cells},
        "joint_pp": 100 * (acc["父∈子∈"] - acc["父∉子∉"]),
        "parent_comp_pp": 100 * (acc["父∈子∈"] - acc["父∉子∈"]),
        "child_comp_pp": 100 * (acc["父∉子∈"] - acc["父∉子∉"]),
        "parent_net_pp": pc.get("gap_pp"),
        "parent_net_pt_pp": pt.get("gap_pp"),
        "overall_acc": (d.get("overall") or {}).get("accuracy"),
    }
    return row, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="组名 + 若干 joint_subgroup_diag*.json；可重复")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    out, problems = {"groups": {}}, []
    L = ["# 父×子二维联合诊断 · 多种子汇总", "",
         "> 口径：仅有子节点的节点（稿件表 4 / 附录表 B3）。A=父∈子∈，C=父∉子∈，D=父∉子∉；"
         "联合差 = A − D = 父链分量 (A − C) + 子链分量 (C − D)。单位 pp；sd 为样本标准差。", "",
         "| 组 | 联合差 | 父链分量 | 子链分量 | 一维父链净效应 | PT 层父链差 |", "| --- | --- | --- | --- | --- | --- |"]
    detail = []
    for g in a.group:
        name, files = g[0], g[1:]
        rows = []
        for f in files:
            if not os.path.isfile(f):
                problems.append("%s：文件不存在 %s" % (name, f))
                continue
            r, why = extract(f)
            if r is None:
                problems.append("%s：%s 已排除（%s）" % (name, os.path.basename(f), why))
                continue
            rows.append(r)
        seeds = [r["train_seed"] for r in rows]
        dup = sorted({s for s in seeds if seeds.count(s) > 1})
        if dup:
            problems.append("%s：训练种子 %s 出现多次（同一 run 被列了两次？）" % (name, dup))
        st = {k: _stats([r[k] for r in rows]) for k in KEYS}
        out["groups"][name] = {"rows": rows, "stats": st}
        L.append("| %s | %s | %s | %s | %s | %s |" % (name, _fmt(st["joint_pp"]), _fmt(st["parent_comp_pp"]),
                                                     _fmt(st["child_comp_pp"]), _fmt(st["parent_net_pp"]),
                                                     _fmt(st["parent_net_pt_pp"])))
        for r in rows:
            detail.append("| %s | %s | %s | %+.2f | %+.2f | %+.2f | %s |" % (
                name, r["file"], r["train_seed"], r["joint_pp"], r["parent_comp_pp"], r["child_comp_pp"],
                "—" if r["parent_net_pp"] is None else "%+.2f" % r["parent_net_pp"]))
    L += ["", "> 稿件表 4（V1，seed 42）：联合差 %+.2f = 父链 %+.2f + 子链 %+.2f；一维父链净效应 %+.2f。" % (
        PAPER_V1["joint_pp"], PAPER_V1["parent_comp_pp"], PAPER_V1["child_comp_pp"], PAPER_V1["parent_net_pp"]), ""]
    L += ["## 逐文件", "", "| 组 | 文件 | 训练种子 | 联合差 | 父链分量 | 子链分量 | 一维父链净效应 |",
          "| --- | --- | --- | --- | --- | --- | --- |"] + detail
    L += ["", "## 问题 / 需要注意", ""] + (["- " + p for p in problems] if problems else ["- 无"])
    out["problems"] = problems
    md = "\n".join(L) + "\n"
    print(md)
    if a.md:
        io.open(a.md, "w", encoding="utf-8", newline="\n").write(md)
        print("written:", a.md)
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="\n").write(json.dumps(out, ensure_ascii=False, indent=1))
        print("written:", a.json)
    return out


if __name__ == "__main__":
    main()
