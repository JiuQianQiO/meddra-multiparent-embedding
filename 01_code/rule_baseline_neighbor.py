# -*- coding: utf-8 -*-
r"""rule_baseline_neighbor.py — E0b 邻居标签规则基线（离线，无 GPU，仅 pandas）。

为什么需要它：
  rule_baseline_soc.py 的 R2–R4 只用拓扑、不用任何标签；但转导式 GNN 训练时**能看到训练集节点的
  SOC 标签**，而 MedDRA 的标签构造使大量测试节点的标签与相邻训练节点逐字相同：
    · LLT 的标签 = 父 PT 的标签；
    · PT 的标签 = 其子 LLT 的标签；
    · HLT/HLGT 的标签 = 子节点标签的多数票。
  因此与模型"信息量相当"的规则基线应当允许读取**训练集节点**的标签（测试节点自身的标签字段仍不可见）。
  这正是审稿人最可能提出的问题：模型的 0.99 是否只是沿层级边复制训练标签。

口径（只读拓扑 + 训练集标签；测试节点自身的 PT_SOC_CODE / PRIMARY_SOC_FG 不可见）：
  R5 nbr_vote     : 1 跳邻居（父 ∪ 子，不含 SOC 节点）中训练集节点标签的多数票；
                    平票时按该节点 SOC 祖先在长表中的出现顺序择一；无已标注邻居时回退 R3。
  R6 child_first  : 级联——训练集子节点多数票 → 训练集父节点多数票 →（仅 LLT）同父 PT 下训练集
                    兄弟 LLT 的多数票 → R3。

输出：总体、分层，以及按 SOC 轴数（单轴 n_SOC=1 / 多轴 n_SOC≥2）分组的 Acc 与 Macro-F1，
      并报告每条规则"有已标注邻居"的覆盖率。R1–R4 一并重算，便于同表对照。

可选 --knn-pred：读入 train_b1_text_embed.py（2026-10-29 补丁版）导出的
  kNN_<model>/test_predictions_kNN_<model>.json，额外报告 kNN 在单轴/多轴子集上的 Acc，
  以及 top-1 近邻与测试节点的层级关系（父 / 子 / 同父兄弟 / 其他）的占比与各自准确率——
  用来检验"kNN 之所以强，是因为最近的训练术语往往就是层级邻居、标签按构造相同"。

用法：python rule_baseline_neighbor.py --csv PATH --split PATH [--md PATH] [--json PATH] [--knn-pred JSON ...]
"""
import argparse
import io
import json
from collections import Counter

from rule_baseline_soc import load, split_level_nodes, build_labels, build_soc_ancestors, macro_f1

LEVELS = ["HLGT", "HLT", "PT", "LLT"]
COL = {"SOC": "SOC_CODE", "HLGT": "HLGT_CODE", "HLT": "HLT_CODE", "PT": "PT_CODE", "LLT": "LLT_CODE"}
PARENT = {"HLGT": "SOC", "HLT": "HLGT", "PT": "HLT", "LLT": "PT"}
_EMPTY = {"parents": frozenset(), "children": frozenset()}


def split_entries(split_path, level_nodes):
    """返回 {'train': [(level, code)], 'val': [...], 'test': [...]}（与 trainer 的 all_nodes 顺序一致）。"""
    sp = json.load(io.open(split_path, encoding="utf-8"))
    flat = [(lv, c) for lv in LEVELS for c in level_nodes[lv]]
    assert [c for _, c in flat] == [int(c) for c in sp["nodes"]], "split nodes 与有序层级节点不一致"
    return {k: [flat[i] for i in sp[k]] for k in ("train", "val", "test")}


def build_adjacency(df):
    """(level, code) -> {'parents': set, 'children': set}；节点为 (level, code)，不含 SOC 层。"""
    adj = {}
    for lv in LEVELS:
        c, pc = COL[lv], COL[PARENT[lv]]
        sub = df.dropna(subset=[c, pc])[[c, pc]].drop_duplicates()
        for code, par in zip(sub[c], sub[pc]):
            node = (lv, int(code))
            adj.setdefault(node, {"parents": set(), "children": set()})
            if PARENT[lv] != "SOC":
                parent = (PARENT[lv], int(par))
                adj[node]["parents"].add(parent)
                adj.setdefault(parent, {"parents": set(), "children": set()})["children"].add(node)
    return adj


def n_soc_map(df):
    out = {}
    for lv in LEVELS:
        g = df.dropna(subset=[COL[lv]]).groupby(COL[lv])["SOC_CODE"].nunique()
        for code, n in g.items():
            out[(lv, int(code))] = int(n)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--knn-pred", nargs="*", default=[],
                    help="B1 kNN 逐节点预测 JSON（含 nn_code / nn_level），可给多个")
    a = ap.parse_args(argv)

    df = load(a.csv)
    level_nodes = split_level_nodes(df)
    labels = build_labels(df)
    ancestors = build_soc_ancestors(df)
    entries = split_entries(a.split, level_nodes)
    adj = build_adjacency(df)
    nsoc = n_soc_map(df)
    test = entries["test"]
    train = set(entries["train"])

    def lab(node):
        return labels[node[0]].get(node[1], -1)

    def soc_order(node):
        return list(dict.fromkeys(ancestors[node[0]].get(node[1], [])))

    def r3(node):
        u = soc_order(node)
        return u[0] if u else -1

    def vote(nodes, node):
        vals = [lab(n) for n in nodes if n in train and lab(n) != -1]
        if not vals:
            return None
        cnt = Counter(vals)
        top = max(cnt.values())
        tied = [v for v, k in cnt.items() if k == top]
        if len(tied) == 1:
            return tied[0]
        for s in soc_order(node):          # 平票：按 SOC 祖先顺序（纯拓扑信息）择一
            if s in tied:
                return s
        return sorted(tied)[0]

    cover = {"R5": 0, "R6": 0}

    def r5(node):
        nb = adj.get(node, _EMPTY)
        v = vote(set(nb["parents"]) | set(nb["children"]), node)
        if v is None:
            return r3(node)
        cover["R5"] += 1
        return v

    def r6(node):
        nb = adj.get(node, _EMPTY)
        for group in (nb["children"], nb["parents"]):
            v = vote(group, node)
            if v is not None:
                cover["R6"] += 1
                return v
        if node[0] == "LLT":
            sibs = set()
            for p in nb["parents"]:
                sibs |= set(adj.get(p, _EMPTY)["children"])
            sibs.discard(node)
            v = vote(sibs, node)
            if v is not None:
                cover["R6"] += 1
                return v
        return r3(node)

    def r2(node):
        u = soc_order(node)
        return u[0] if len(u) == 1 else -1

    def r4(node):
        a_ = ancestors[node[0]].get(node[1], [])
        return Counter(a_).most_common(1)[0][0] if a_ else -1

    rules = [
        ("R1", "label_lookup（测试节点自身标签字段，上限参照）", lab),
        ("R2", "topo_unique（多个 SOC 祖先时弃权）", r2),
        ("R3", "topo_first", r3),
        ("R4", "topo_majority", r4),
        ("R5", "nbr_vote（训练集标签，1 跳）", r5),
        ("R6", "child_first（训练集标签，级联）", r6),
    ]
    yt = [lab(n) for n in test]
    groups = [("all", list(range(len(test)))),
              ("single-axis (n_SOC=1)", [i for i, n in enumerate(test) if nsoc.get(n, 1) == 1]),
              ("multiaxial (n_SOC>=2)", [i for i, n in enumerate(test) if nsoc.get(n, 1) >= 2])]
    for lv in LEVELS:
        groups.append((lv, [i for i, n in enumerate(test) if n[0] == lv]))
        groups.append((lv + " multiaxial", [i for i, n in enumerate(test) if n[0] == lv and nsoc.get(n, 1) >= 2]))

    preds = {rid: [fn(n) for n in test] for rid, _, fn in rules}
    result = {"n_test": len(test), "rules": {rid: desc for rid, desc, _ in rules}, "metrics": {}}
    lines = ["# E0b 规则基线（含训练集邻居标签）", "",
             "> 数据：`%s`；划分：`%s`；n_test=%d。" % (a.csv, a.split, len(test)),
             "> R1 读取测试节点自身的标签字段（上限参照）；R2–R4 只用拓扑；R5–R6 只用拓扑 + **训练集**节点标签，"
             "与转导式 GNN 可见的信息一致。R2 的弃权计为错误。", ""]
    for rid, desc, _ in rules:
        lines.append("- **%s** %s" % (rid, desc))
    lines += ["", "| 子集 | n | " + " | ".join("%s Acc / MF1" % rid for rid, _, _ in rules) + " |",
              "| --- | --- | " + " | ".join("---" for _ in rules) + " |"]
    for g, ix in groups:
        if not ix:
            continue
        cells = []
        t = [yt[i] for i in ix]
        for rid, _, _ in rules:
            p = [preds[rid][i] for i in ix]
            acc = sum(1 for x, y in zip(t, p) if x == y) / len(t)
            mf1 = macro_f1(t, p)
            result["metrics"].setdefault(rid, {})[g] = {"n": len(t), "acc": acc, "macro_f1": mf1}
            cells.append("%.4f / %.4f" % (acc, mf1))
        lines.append("| %s | %d | %s |" % (g, len(ix), " | ".join(cells)))
    lines += ["", "覆盖率（测试节点中有 ≥1 个已标注邻居、未回退到 R3 的比例）：",
              "- R5：%d / %d = %.2f%%" % (cover["R5"], len(test), 100.0 * cover["R5"] / len(test)),
              "- R6：%d / %d = %.2f%%" % (cover["R6"], len(test), 100.0 * cover["R6"] / len(test)), ""]
    result["coverage"] = {k: v / len(test) for k, v in cover.items()}

    # ---- 可选：kNN 近邻关系分析 ----
    def relation(node, nn):
        nb_ = adj.get(node, _EMPTY)
        if nn in nb_["parents"]:
            return "parent"
        if nn in nb_["children"]:
            return "child"
        for p in nb_["parents"]:
            if nn in adj.get(p, _EMPTY)["children"]:
                return "sibling"
        return "other"

    for kp in a.knn_pred:
        recs = json.load(io.open(kp, encoding="utf-8"))
        rows, n_skip = [], 0
        for r in recs:
            node = (str(r["level"]).upper(), int(r["node_code"]))
            if node not in adj or "nn_code" not in r:
                n_skip += 1
                continue
            nn = (str(r["nn_level"]).upper(), int(r["nn_code"]))
            ok = int(r["true_soc_code"]) == int(r["pred_soc_code"])
            rows.append((node, relation(node, nn), ok, nsoc.get(node, 1) >= 2))
        if not rows:
            lines.append("- %s：没有可用记录（缺 nn_code 字段或节点口径不一致），跳过" % kp)
            continue
        kres = {"n": len(rows), "skipped": n_skip, "by_axis": {}, "by_relation": {}}
        lines += ["", "## kNN 近邻分析：`%s`" % kp, "",
                  "| 子集 | n | 占比 | kNN Acc |", "| --- | --- | --- | --- |"]
        for name, sel in (("all", lambda x: True),
                          ("single-axis (n_SOC=1)", lambda x: not x[3]),
                          ("multiaxial (n_SOC>=2)", lambda x: x[3])):
            sub = [x for x in rows if sel(x)]
            if sub:
                acc = sum(x[2] for x in sub) / len(sub)
                kres["by_axis"][name] = {"n": len(sub), "acc": acc}
                lines.append("| %s | %d | %.2f%% | %.4f |" % (name, len(sub), 100.0 * len(sub) / len(rows), acc))
        for rel in ("parent", "child", "sibling", "other"):
            sub = [x for x in rows if x[1] == rel]
            if sub:
                acc = sum(x[2] for x in sub) / len(sub)
                kres["by_relation"][rel] = {"n": len(sub), "share": len(sub) / len(rows), "acc": acc}
                lines.append("| top-1 近邻 = %s | %d | %.2f%% | %.4f |" % (rel, len(sub), 100.0 * len(sub) / len(rows), acc))
        if n_skip:
            lines.append("")
            lines.append("（%d 条记录因缺字段或节点口径不一致被跳过）" % n_skip)
        result.setdefault("knn", {})[kp] = kres

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
