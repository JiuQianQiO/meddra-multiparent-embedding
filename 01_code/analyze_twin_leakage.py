# -*- coding: utf-8 -*-
r"""analyze_twin_leakage.py — 孪生 LLT 泄漏量化（本地，无 GPU）。

审稿意见 M3：MedDRA 每个 PT 都有一个同码、同名的 LLT，随机划分下测试 PT 的同名 LLT
约有 81% 落在训练集，两者特征与标签相同且经反向边直连，可能系统性抬高转导结果。

本脚本把「测试 PT 按同名 LLT 是否在训练集」分两组，报告两组的样本量与准确率，
作为该通道强度的直接证据。输入全部可离线获得：

用法：
  python analyze_twin_leakage.py \
      --predictions D:\MedDRA模型文件\JBI\analysis_out\test_predictions_SAGE.json \
      --split       D:\MedDRA模型文件\JBI\split_indices.json \
      --csv         D:\MedDRAData\MEDDRA_DATA_29_0.csv

输出两组（孪生 LLT 在/不在训练集）的 n、正确数、Acc，以及「若两组无差，泄漏不成立」的
简单对照。同时给出反向通道（测试 LLT 的同名 PT 是否在训练集）。
"""
import argparse
import csv
import io
import json
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    pred = { (p["node_code"], p["level"]): p["correct"] for p in json.load(io.open(args.predictions, encoding="utf-8")) }
    sp = json.load(io.open(args.split, encoding="utf-8"))
    nodes = sp["nodes"]; train = set(sp["train"]); test = set(sp["test"])
    N = len(nodes)
    # 训练器 all_nodes 顺序：HLGT, HLT, PT, LLT（SOC 已跳过）
    HLGT = (0, 337); HLT = (337, 337 + 1739)
    PT = (337 + 1739, 337 + 1739 + 27361); LLT = (337 + 1739 + 27361, N)

    code2idx = defaultdict(list)
    for i, c in enumerate(nodes):
        code2idx[int(c)].append(i)

    def group_report(indices, label, match_interval):
        in_train = []; not_train = []
        for i in sorted(indices):
            c = int(nodes[i])
            twins = [j for j in code2idx[c] if match_interval[0] <= j < match_interval[1]]
            # 同名节点（同 code）在目标区间的索引里是否有一个在训练集
            if any(j in train for j in twins):
                in_train.append(i)
            else:
                not_train.append(i)
        def acc(idx_list):
            corr = sum(1 for i in idx_list if pred.get((int(nodes[i]), label), False))
            return len(idx_list), corr, (corr / len(idx_list) if idx_list else float("nan"))
        n1, c1, a1 = acc(in_train)
        n2, c2, a2 = acc(not_train)
        return n1, c1, a1, n2, c2, a2

    # 正向通道：测试 PT 的同名 LLT 是否在训练集
    test_pt = sorted(i for i in range(*PT) if i in test)
    n1, c1, a1, n2, c2, a2 = group_report(test_pt, "pt", LLT)
    print("=" * 80)
    print("正向通道：测试 PT 节点（%d 个）按「同名 LLT 是否在训练集」分组" % len(test_pt))
    print("  同名 LLT 在训练集    : n=%4d  correct=%4d  Acc=%.4f" % (n1, c1, a1))
    print("  同名 LLT 不在训练集  : n=%4d  correct=%4d  Acc=%.4f" % (n2, c2, a2))
    print("  两组 Acc 差          : %.4f（>0 表示存在孪生泄漏通道）" % (a1 - a2))

    # 反向通道：测试 LLT 的同名 PT 是否在训练集
    test_llt = sorted(i for i in range(*LLT) if i in test)
    m1, k1, b1, m2, k2, b2 = group_report(test_llt, "llt", PT)
    print("=" * 80)
    print("反向通道：测试 LLT 节点（%d 个）按「同名 PT 是否在训练集」分组" % len(test_llt))
    print("  同名 PT 在训练集    : n=%4d  correct=%4d  Acc=%.4f" % (m1, k1, b1))
    print("  同名 PT 不在训练集  : n=%4d  correct=%4d  Acc=%.4f" % (m2, k2, b2))
    print("  两组 Acc 差          : %.4f" % (b1 - b2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
