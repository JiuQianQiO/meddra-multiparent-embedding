# -*- coding: utf-8 -*-
"""有监督图基线：Label Propagation（LP）与 Correct & Smooth（C&S，平滑部分）。

回答审稿意见 M6-b："缺少 LP/C&S 等有监督图基线"。
LP 只用图结构 + 训练标签（与规则基线 R6 互补，是经典半监督分类方法）；
C&S-Smooth 在 LP 的软预测上做标签平滑（不依赖额外特征）。

说明：主实验的转导协议里，四层节点（HLGT/HLT/PT/LLT，共 120,519）与 27 个 SOC 节点
共同构成图；SOC 不参与监督但作为图中的节点参与传播。本脚本与主实验一致：
在 120,546 节点图上传播，仅在四层节点的测试集上评估。

用法（需先配置好依赖，复用 train_b2_kge 的处理器）：
    python lp_cs_baseline.py \
        --csv D:/MedDRAData/MEDDRA_DATA_29_0.csv \
        --cache-dir D:/MedDRA模型文件/JBI/cache_shared \
        --out-dir D:/MedDRA模型文件/JBI/Baseline \
        --lp-iters 50 --cs-alpha 0.8 --cs-iters 20
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import scipy.sparse as sp_sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _normalized_adjacency(graph, exclude=("soc",)):
    """从 graph 的边构建无向、加自环、行归一化的邻接矩阵，排除指定节点类型（默认 SOC）。

    SOC 节点是"类别名"（SOC 名称=类别），不应作为标签传播的载体：若把 SOC 纳入，
    其无监督的标签会污染四层节点的标签传播。故仅在四层节点（HLGT/HLT/PT/LLT）上构建。
    """
    node_types = [nt for nt in graph.node_types if nt not in exclude]
    starts, acc = {}, 0
    for nt in node_types:
        starts[nt] = acc
        acc += len(graph[nt].node_codes)
    n = acc
    rows, cols = [], []
    for (src, rel, dst) in graph.edge_types:
        if src in exclude or dst in exclude:
            continue
        ei = graph[(src, rel, dst)].edge_index
        h = ei[0].numpy() + starts[src]
        t = ei[1].numpy() + starts[dst]
        rows += list(h) + list(t)
        cols += list(t) + list(h)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.ones(len(rows), dtype=np.float32)
    A = sp_sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A = A + sp_sparse.eye(n, format="csr")          # 自环
    deg = np.asarray(A.sum(1)).ravel()
    d_inv = 1.0 / np.maximum(deg, 1e-12)
    D_inv = sp_sparse.diags(d_inv)
    return (D_inv @ A).tocsr()                      # 行归一化 D^-1 A（每行和为1）


def label_propagation(adj, labels, train_mask, iters):
    """标签传播：训练节点固定为真实标签，其余节点按邻接聚合。"""
    y = labels.astype(np.float32).copy()
    y[~train_mask] = 0.0
    for _ in range(iters):
        y = adj @ y
        y[train_mask] = labels[train_mask].astype(np.float32)
    return y


def smooth(adj, base_preds, labels, train_mask, alpha, iters):
    """Correct & Smooth 的平滑步：软预测在图上做标签平滑，训练节点保持硬标签。"""
    y = base_preds.astype(np.float32).copy()
    y[train_mask] = labels[train_mask].astype(np.float32)
    for _ in range(iters):
        y = (1.0 - alpha) * y + alpha * (adj @ y)
        y[train_mask] = labels[train_mask].astype(np.float32)
    return y


def _macro_f1(pred, true, n_class):
    from sklearn.metrics import f1_score
    return f1_score(true, pred, labels=list(range(n_class)), average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", default="./lp_cs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lp-iters", type=int, default=50)
    ap.add_argument("--cs-alpha", type=float, default=0.8)
    ap.add_argument("--cs-iters", type=int, default=20)
    args = ap.parse_args()

    os.environ["MEDDRA_CSV_PATH"] = args.csv
    os.environ["MEDDRA_CACHE_DIR"] = args.cache_dir
    os.makedirs(args.out_dir, exist_ok=True)

    import train_b2_kge as b2
    cfg = b2.BaselineConfig(csv_path=args.csv, cache_dir=args.cache_dir, seed=args.seed)
    proc, graph = b2.build_processor(cfg)
    sp = b2.get_split(proc, graph, seed=args.seed)
    y = sp["y"]                                   # 四层节点的 SOC 标签（0..26）
    tr, va, te = sp["train"], sp["val"], sp["test"]
    n_class = 27

    # 四层节点邻接（排除 SOC，避免 SOC"类别名"污染标签传播）
    adj = _normalized_adjacency(graph)

    # 标签矩阵：仅四层节点（fold_n == len(y)），one-hot
    fold_n = len(y)
    labels = np.zeros((fold_n, n_class), dtype=np.float32)
    labels[np.arange(fold_n), y] = 1.0
    train_mask = np.zeros(fold_n, dtype=bool)
    train_mask[tr] = True

    results = {}
    # ---- LP ----
    lp = label_propagation(adj, labels, train_mask, args.lp_iters)
    lp_pred = lp.argmax(1)
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        acc = (lp_pred[idx] == y[idx]).mean()
        mf1 = _macro_f1(lp_pred[idx], y[idx], n_class)
        results[f"LP_{name}_acc"] = float(acc)
        results[f"LP_{name}_macro_f1"] = float(mf1)
        print(f"LP   {name:5s}: Acc={acc:.4f} Macro-F1={mf1:.4f}")

    # ---- C&S-Smooth（以 LP 的软预测为 base，做标签平滑）----
    cs = smooth(adj, lp, labels, train_mask, args.cs_alpha, args.cs_iters)
    cs_pred = cs.argmax(1)
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        acc = (cs_pred[idx] == y[idx]).mean()
        mf1 = _macro_f1(cs_pred[idx], y[idx], n_class)
        results[f"CS_{name}_acc"] = float(acc)
        results[f"CS_{name}_macro_f1"] = float(mf1)
        print(f"C&S  {name:5s}: Acc={acc:.4f} Macro-F1={mf1:.4f}")

    outp = os.path.join(args.out_dir, "lp_cs_baseline.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"结果已保存: {outp}")


if __name__ == "__main__":
    main()
