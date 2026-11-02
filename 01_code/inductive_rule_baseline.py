# -*- coding: utf-8 -*-
r"""inductive_rule_baseline.py — 归纳式（留出 HLT 子树）协议下的规则基线（离线，无 GPU，仅 pandas）。

为什么需要它：
  转导式测试里，测试节点的标签往往能从训练集邻居直接抄来（见 rule_baseline_neighbor.py）。
  归纳式 KEEP 协议把 174 棵 HLT 子树整体留出，子树内所有节点的标签都不可见，但层级边保留——
  留出节点仍能连到子树外**已标注的祖先**（HLGT，以及多父 PT 的其他父 HLT）。
  与 IND-SAGE-KEEP（0.9333 / 0.8682）信息量相当的规则是"沿父边向上，找最近的已标注祖先投票"。
  它决定正文情况 B 那句"归纳式下模型优于最近祖先标签规则"能否成立。

口径：
  · 留出集 = inductive_holdout.json 中 holdout_node_codes 覆盖的全部 (层级, 代码) 实例
    （与训练器一致：按代码判断，PT 与同码 LLT 一起留出；论文为 HLT 174 / PT 4,157 / LLT 14,236 = 18,567）。
  · 已标注 = 其余全部非 SOC 节点（归纳式训练集）。
  · I3 topo_first       ：只读拓扑，取首个 SOC 祖先（同 rule_baseline_soc.py 的 R3）。
  · I4 topo_majority    ：只读拓扑，路径数多数（同 R4）。
  · I5 nearest_labelled ：逐跳向上找父节点，第一层出现已标注祖先即对其标签投票；
                          平票按 SOC 祖先顺序择一；找不到时回退 I3。
  · 另报告 I5 在哪一跳命中（1 = 直接父节点已标注，例如多父 PT 的另一个 HLT；2 = 祖父……）。

用法：
  python inductive_rule_baseline.py --csv MEDDRA_DATA_29_0.csv \
    --holdout <OUTPUT_ROOT>/IND-SAGE-KEEP/inductive_holdout.json \
    [--md INDUCTIVE_RULE_BASELINE.md] [--json INDUCTIVE_RULE_BASELINE.json]
"""
import argparse
import io
import json
from collections import Counter

from rule_baseline_soc import load, split_level_nodes, build_labels, build_soc_ancestors, macro_f1
from rule_baseline_neighbor import build_adjacency, n_soc_map

LEVELS = ["HLGT", "HLT", "PT", "LLT"]
_EMPTY = {"parents": frozenset(), "children": frozenset()}

# 稿件表 4（单次运行）中的模型参照值，仅用于并排展示
MODEL_REF = {
    "IND-SAGE-KEEP": {"acc": 0.9333, "macro_f1": 0.8682, "HLT": 0.9253, "PT": 0.9310, "LLT": 0.9340},
    "IND-GAT-KEEP": {"acc": 0.9315, "macro_f1": 0.8617, "HLT": 0.9195, "PT": 0.9298, "LLT": 0.9321},
    "IND-noGNN (PubMedBERT)": {"acc": 0.7453, "macro_f1": 0.6299, "HLT": 0.8276, "PT": 0.7640, "LLT": 0.7388},
}
EXPECTED = {"HLT": 174, "PT": 4157, "LLT": 14236}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--holdout", required=True, help="IND-*-KEEP 运行目录下的 inductive_holdout.json")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    df = load(a.csv)
    level_nodes = split_level_nodes(df)
    labels = build_labels(df)
    ancestors = build_soc_ancestors(df)
    adj = build_adjacency(df)
    nsoc = n_soc_map(df)
    man = json.load(io.open(a.holdout, encoding="utf-8"))
    codes = {int(c) for c in man.get("holdout_node_codes", [])}
    if not codes:
        raise SystemExit("留出清单为空：%s" % a.holdout)

    universe = [(lv, int(c)) for lv in LEVELS for c in level_nodes[lv]]
    held = [n for n in universe if n[1] in codes]
    held_set = set(held)
    labelled = {n for n in universe if n not in held_set}
    by_lv = Counter(n[0] for n in held)
    warn = []
    for lv, exp in EXPECTED.items():
        if by_lv.get(lv, 0) != exp:
            warn.append("%s 留出实例数 %d 与论文 %d 不一致" % (lv, by_lv.get(lv, 0), exp))
    if by_lv.get("HLGT"):
        warn.append("有 %d 个 HLGT 实例的代码落在留出清单中（代码与其他层撞号？）" % by_lv["HLGT"])

    def lab(node):
        return labels[node[0]].get(node[1], -1)

    def soc_order(node):
        return list(dict.fromkeys(ancestors[node[0]].get(node[1], [])))

    def i3(node):
        u = soc_order(node)
        return u[0] if u else -1

    def i4(node):
        a_ = ancestors[node[0]].get(node[1], [])
        return Counter(a_).most_common(1)[0][0] if a_ else -1

    hop_hist = Counter()

    def i5(node):
        frontier = set(adj.get(node, _EMPTY)["parents"])
        seen = set(frontier)
        hop = 1
        while frontier:
            vals = [lab(n) for n in frontier if n in labelled and lab(n) != -1]
            if vals:
                hop_hist[hop] += 1
                cnt = Counter(vals)
                top = max(cnt.values())
                tied = [v for v, k in cnt.items() if k == top]
                if len(tied) == 1:
                    return tied[0]
                for s in soc_order(node):
                    if s in tied:
                        return s
                return sorted(tied)[0]
            nxt = set()
            for n in frontier:
                nxt |= set(adj.get(n, _EMPTY)["parents"])
            frontier = nxt - seen
            seen |= frontier
            hop += 1
        hop_hist["fallback"] += 1
        return i3(node)

    rules = [("I3", "topo_first（只读拓扑）", i3),
             ("I4", "topo_majority（只读拓扑）", i4),
             ("I5", "nearest_labelled（最近已标注祖先投票）", i5)]
    yt = [lab(n) for n in held]
    preds = {rid: [fn(n) for n in held] for rid, _, fn in rules}
    groups = [("all", list(range(len(held)))),
              ("single-axis (n_SOC=1)", [i for i, n in enumerate(held) if nsoc.get(n, 1) == 1]),
              ("multiaxial (n_SOC>=2)", [i for i, n in enumerate(held) if nsoc.get(n, 1) >= 2])]
    for lv in ("HLT", "PT", "LLT"):
        groups.append((lv, [i for i, n in enumerate(held) if n[0] == lv]))

    result = {"holdout_file": a.holdout, "n_holdout_codes": len(codes), "n_holdout_instances": len(held),
              "by_level": dict(by_lv), "warnings": warn, "metrics": {}, "model_reference": MODEL_REF}
    lines = ["# 归纳式（KEEP）规则基线（离线复算）", "",
             "> 留出清单：`%s`；留出代码 %d 个，(层级, 代码) 实例 %d 个（%s）。" % (
                 a.holdout, len(codes), len(held), ", ".join("%s %d" % (lv, by_lv.get(lv, 0)) for lv in ("HLT", "PT", "LLT"))),
             "> 已标注节点 = 留出子树之外的全部非 SOC 节点；留出节点自身的标签字段不可见。", ""]
    for w in warn:
        lines.append("> ⚠️ %s" % w)
    if warn:
        lines.append("")
    for rid, desc, _ in rules:
        lines.append("- **%s** %s" % (rid, desc))
    lines += ["", "| 子集 | n | " + " | ".join("%s Acc / MF1" % rid for rid, _, _ in rules) + " |",
              "| --- | --- | " + " | ".join("---" for _ in rules) + " |"]
    for g, ix in groups:
        if not ix:
            continue
        t = [yt[i] for i in ix]
        cells = []
        for rid, _, _ in rules:
            p = [preds[rid][i] for i in ix]
            acc = sum(1 for x, y in zip(t, p) if x == y) / len(t)
            mf1 = macro_f1(t, p)
            result["metrics"].setdefault(rid, {})[g] = {"n": len(t), "acc": acc, "macro_f1": mf1}
            cells.append("%.4f / %.4f" % (acc, mf1))
        lines.append("| %s | %d | %s |" % (g, len(ix), " | ".join(cells)))
    total = sum(hop_hist.values()) or 1
    result["i5_hop_hist"] = {str(k): v for k, v in hop_hist.items()}
    lines += ["", "I5 命中跳数分布：" + "，".join(
        "%s 跳 %d（%.1f%%）" % (k, v, 100.0 * v / total) for k, v in sorted(hop_hist.items(), key=lambda kv: str(kv[0]))), "",
        "## 稿件表 4 的模型参照值（单次运行）", "",
        "| 模型 | Acc | Macro-F1 | HLT Acc | PT Acc | LLT Acc |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in MODEL_REF.items():
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f |" % (k, v["acc"], v["macro_f1"], v["HLT"], v["PT"], v["LLT"]))
    i5_all = result["metrics"]["I5"]["all"]
    gap = MODEL_REF["IND-SAGE-KEEP"]["acc"] - i5_all["acc"]
    result["gap_ind_sage_keep_minus_i5_acc"] = gap
    lines += ["", "IND-SAGE-KEEP − I5 = **%+.2f** 个百分点（Acc）。" % (100 * gap), ""]
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
