# 正文修订稿所需数字（自动汇总）

## 1. E0 结论句：情况 B

- V1 − R6 = +0.23 pp（未达到 ≥ 1 pp 且超出 V1 ±2 sd 的标准，2 sd = 0.11 pp）

## 2. 占位符

| 占位符 | 值 | 来源 |
| --- | --- | --- |
| R5 Acc/MF1 | 0.9610/0.9421 | RULE_BASELINE_NEIGHBOR.json |
| R5 single-axis Acc/MF1 | 0.9989/0.9975 | RULE_BASELINE_NEIGHBOR.json |
| R5 multiaxial Acc/MF1 | 0.9143/0.8575 | RULE_BASELINE_NEIGHBOR.json |
| R6 Acc/MF1 | 0.9917/0.9883 | RULE_BASELINE_NEIGHBOR.json |
| R6 single-axis Acc/MF1 | 0.9992/0.9982 | RULE_BASELINE_NEIGHBOR.json |
| R6 multiaxial Acc/MF1 | 0.9824/0.9738 | RULE_BASELINE_NEIGHBOR.json |
| R6 coverage | 97.53 | RULE_BASELINE_NEIGHBOR.json |
| strongest rule (no test-node label fields) | R6 0.9917 | RULE_BASELINE_NEIGHBOR.json |
| kNN-BGE share_hier | 76.89 | RULE_BASELINE_NEIGHBOR.json --knn-pred |
| kNN-BGE share_child | 20.02 | 同上 |
| kNN-BGE share_sib | 31.33 | 同上 |
| kNN-BGE share_parent | 25.54 | 同上 |
| kNN-BGE acc_hier | 0.9979 | 同上 |
| kNN-BGE acc_other | 0.7352 | 同上 |
| kNN-BGE single-axis | 0.9422 | 同上 |
| kNN-BGE multiaxial | 0.9311 | 同上 |
| kNN-E5 share_hier | 74.06 | RULE_BASELINE_NEIGHBOR.json --knn-pred |
| kNN-E5 share_child | 19.84 | 同上 |
| kNN-E5 share_sib | 29.67 | 同上 |
| kNN-E5 share_parent | 24.56 | 同上 |
| kNN-E5 acc_hier | 0.9983 | 同上 |
| kNN-E5 acc_other | 0.7134 | 同上 |
| kNN-E5 single-axis | 0.9318 | 同上 |
| kNN-E5 multiaxial | 0.9154 | 同上 |
| kNN-BGE Acc/MF1 | 0.9372/0.9168 | result_B1_bge.json |
| kNN-E5 Acc/MF1 | 0.9244/0.8987 | result_B1_e5.json |
| ΔMF1 vs kNN-BGE (paper V1) | 7.24 | 0.9892 − kNN-BGE Macro-F1 |
| ΔMF1 vs kNN-BGE (V1 mean) | 7.39 | V1 n=5 均值 |
| Δacc vs kNN-BGE (paper V1) | 5.60 | 0.9932 − kNN-BGE Acc |
| V1 single-axis Acc | 0.9981 ± 0.0001 (n=5) | MULTIAXIAL_SUMMARY.json |
| V1 multiaxial Acc | 0.9889 ± 0.0014 (n=5) | 同上 |
| V1 multiaxial MF1 | 0.9813 ± 0.0028 (n=5) | 同上 |
| V1 err_share_multi | 0.8273 ± 0.0261 (n=5) | 同上 |
| n_err (V1) | 72.6 | 同上（各种子均值） |
| n_err_multi (V1) | 60.2 | 同上（各种子均值） |
| V2 single-axis Acc | 0.9982 ± 0.0002 (n=4) | MULTIAXIAL_SUMMARY.json |
| V2 multiaxial Acc | 0.9886 ± 0.0024 (n=4) | 同上 |
| V2 multiaxial MF1 | 0.9810 ± 0.0037 (n=4) | 同上 |
| V2 err_share_multi | 0.8299 ± 0.0392 (n=4) | 同上 |
| C7 single-axis Acc | 0.9978 ± 0.0002 (n=3) | MULTIAXIAL_SUMMARY.json |
| C7 multiaxial Acc | 0.9711 ± 0.0035 (n=3) | 同上 |
| C7 multiaxial MF1 | 0.9451 ± 0.0077 (n=3) | 同上 |
| C7 err_share_multi | 0.9137 ± 0.0064 (n=3) | 同上 |
| V3 single-axis Acc | 0.9970 ± 0.0003 (n=4) | MULTIAXIAL_SUMMARY.json |
| V3 multiaxial Acc | 0.9640 ± 0.0115 (n=4) | 同上 |
| V3 multiaxial MF1 | 0.9344 ± 0.0188 (n=4) | 同上 |
| V3 err_share_multi | 0.9020 ± 0.0293 (n=4) | 同上 |
| V4 single-axis Acc | 0.9974 ± 0.0001 (n=4) | MULTIAXIAL_SUMMARY.json |
| V4 multiaxial Acc | 0.9768 ± 0.0068 (n=4) | 同上 |
| V4 multiaxial MF1 | 0.9591 ± 0.0128 (n=4) | 同上 |
| V4 err_share_multi | 0.8732 ± 0.0316 (n=4) | 同上 |
| E1A single-axis Acc | 0.9982 ± 0.0002 (n=3) | MULTIAXIAL_SUMMARY.json |
| E1A multiaxial Acc | 0.9893 ± 0.0027 (n=3) | 同上 |
| E1A multiaxial MF1 | 0.9827 ± 0.0039 (n=3) | 同上 |
| E1A err_share_multi | 0.8292 ± 0.0195 (n=3) | 同上 |
| E1B single-axis Acc | 0.9982 ± 0.0000 (n=3) | MULTIAXIAL_SUMMARY.json |
| E1B multiaxial Acc | 0.9877 ± 0.0007 (n=3) | 同上 |
| E1B multiaxial MF1 | 0.9802 ± 0.0019 (n=3) | 同上 |
| E1B err_share_multi | 0.8465 ± 0.0080 (n=3) | 同上 |
| SOCDROP single-axis Acc | 0.9980 ± 0.0002 (n=3) | MULTIAXIAL_SUMMARY.json |
| SOCDROP multiaxial Acc | 0.9876 ± 0.0018 (n=3) | 同上 |
| SOCDROP multiaxial MF1 | 0.9790 ± 0.0018 (n=3) | 同上 |
| SOCDROP err_share_multi | 0.8318 ± 0.0308 (n=3) | 同上 |
| kNN-BGE single-axis Acc | 0.9422 (n=1) | MULTIAXIAL_SUMMARY.json |
| kNN-BGE multiaxial Acc | 0.9311 (n=1) | 同上 |
| kNN-BGE multiaxial MF1 | 0.7925 (n=1) | 同上 |
| kNN-BGE err_share_multi | 0.4921 (n=1) | 同上 |
| kNN-E5 single-axis Acc | 0.9318 (n=1) | MULTIAXIAL_SUMMARY.json |
| kNN-E5 multiaxial Acc | 0.9154 (n=1) | 同上 |
| kNN-E5 multiaxial MF1 | 0.7756 (n=1) | 同上 |
| kNN-E5 err_share_multi | 0.5022 (n=1) | 同上 |
| inductive ancestor-label rule (I5) Acc/MF1 | 0.8172/0.7167 | INDUCTIVE_RULE_BASELINE.json |
| IND-SAGE-KEEP − I5 (Acc) | 11.61 | 同上（模型值取自稿件表 4） |
| Δ overall (V1 − R6, Acc) | 0.23 | V1 n=5 均值 − R6 |
| Δ multiaxial (V1 − R6, Acc) | 0.64 | MULTIAXIAL_SUMMARY − R6 multiaxial |

## 3. SOC 控制（分层，mean ± sd；Δ 相对 V1 均值）

> 3 层 GNN 下 LLT 看不到 SOC 节点，以 PT（n=2,738）为主要读数。`!` 表示 |Δ| 超过 V1 的 2 sd。

| 组 | 指标 | HLGT | HLT | PT | LLT | 总体 |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | Acc | 0.9429 ± 0.0000 (n=5) | 0.9412 ± 0.0051 (n=5) | 0.9869 ± 0.0014 (n=5) | 0.9973 ± 0.0005 (n=5) | 0.9940 ± 0.0006 (n=5) |
| V1 | Macro-F1 | 0.8874 ± 0.0202 (n=5) | 0.8995 ± 0.0030 (n=5) | 0.9772 ± 0.0029 (n=5) | 0.9958 ± 0.0011 (n=5) | 0.9907 ± 0.0011 (n=5) |
| E1A | Acc | 0.9429 ± 0.0000 (n=3)（-0.00 pp） | 0.9454 ± 0.0086 (n=3)（+0.41 pp） | 0.9882 ± 0.0012 (n=3)（+0.13 pp） | 0.9972 ± 0.0012 (n=3)（-0.01 pp） | 0.9942 ± 0.0013 (n=3)（+0.03 pp） |
| E1A | Macro-F1 | 0.8784 ± 0.0000 (n=3)（-0.90 pp） | 0.9098 ± 0.0081 (n=3)（+1.03 pp!） | 0.9803 ± 0.0010 (n=3)（+0.31 pp） | 0.9959 ± 0.0018 (n=3)（+0.00 pp） | 0.9914 ± 0.0016 (n=3)（+0.07 pp） |
| E1B | Acc | 0.9333 ± 0.0165 (n=3)（-0.95 pp!） | 0.9454 ± 0.0033 (n=3)（+0.41 pp） | 0.9864 ± 0.0006 (n=3)（-0.05 pp） | 0.9968 ± 0.0004 (n=3)（-0.05 pp） | 0.9935 ± 0.0003 (n=3)（-0.05 pp） |
| E1B | Macro-F1 | 0.8729 ± 0.0493 (n=3)（-1.45 pp） | 0.9151 ± 0.0052 (n=3)（+1.56 pp!） | 0.9773 ± 0.0010 (n=3)（+0.00 pp） | 0.9952 ± 0.0008 (n=3)（-0.06 pp） | 0.9904 ± 0.0007 (n=3)（-0.03 pp） |
| SOCDROP | Acc | 0.9143 ± 0.0000 (n=3)（-2.86 pp!） | 0.9416 ± 0.0086 (n=3)（+0.04 pp） | 0.9861 ± 0.0022 (n=3)（-0.07 pp） | 0.9968 ± 0.0005 (n=3)（-0.05 pp） | 0.9933 ± 0.0007 (n=3)（-0.06 pp） |
| SOCDROP | Macro-F1 | 0.8324 ± 0.0157 (n=3)（-5.50 pp!） | 0.9176 ± 0.0187 (n=3)（+1.81 pp!） | 0.9762 ± 0.0029 (n=3)（-0.10 pp） | 0.9953 ± 0.0005 (n=3)（-0.05 pp） | 0.9901 ± 0.0006 (n=3)（-0.06 pp） |

## 4. 多种子主表（总体 Acc / Macro-F1）

| 组 | Acc | Macro-F1 | ΔMF1 vs V1 | 与阈值 / V1 ±2 sd 的比较 |
| --- | --- | --- | --- | --- |
| V1 | 0.9940 ± 0.0006 (n=5) | 0.9907 ± 0.0011 (n=5) | — | — |
| V2 | 0.9939 ± 0.0009 (n=5) | 0.9909 ± 0.0010 (n=5) | +0.02 pp | 在 0.93 pp 阈值内；在 V1 ±2 sd 内 |
| C7 | 0.9854 ± 0.0019 (n=5) | 0.9769 ± 0.0029 (n=5) | -1.38 pp | 超过 0.93 pp 阈值；超出 V1 ±2 sd |
| C3 | 0.9903 (n=1) | 0.9846 (n=1) | -0.60 pp | 在 0.93 pp 阈值内；超出 V1 ±2 sd |
| V3 | 0.9807 ± 0.0056 (n=5) | 0.9692 ± 0.0094 (n=5) | -2.15 pp | 超过 0.93 pp 阈值；超出 V1 ±2 sd |
| V4 | 0.9883 ± 0.0027 (n=5) | 0.9817 ± 0.0042 (n=5) | -0.89 pp | 在 0.93 pp 阈值内；超出 V1 ±2 sd |
| E1A | 0.9942 ± 0.0013 (n=3) | 0.9914 ± 0.0016 (n=3) | +0.07 pp | 在 0.93 pp 阈值内；在 V1 ±2 sd 内 |
| E1B | 0.9935 ± 0.0003 (n=3) | 0.9904 ± 0.0007 (n=3) | -0.03 pp | 在 0.93 pp 阈值内；在 V1 ±2 sd 内 |
| SOCDROP | 0.9933 ± 0.0007 (n=3) | 0.9901 ± 0.0006 (n=3) | -0.06 pp | 在 0.93 pp 阈值内；在 V1 ±2 sd 内 |
| NGPB | 0.8336 ± 0.0025 (n=3) | 0.7785 ± 0.0033 (n=3) | -21.22 pp | 超过 0.93 pp 阈值；超出 V1 ±2 sd |
| GAT | 0.9919 ± 0.0008 (n=3) | 0.9884 ± 0.0013 (n=3) | -0.23 pp | 在 0.93 pp 阈值内；超出 V1 ±2 sd |

## 4b. final_submission 稿件占位符（键名与稿件中黄色 [[...]] 一致）

> 稿件里另有两处条件块：`[[CASE A — …]]` / `[[CASE B — …]]` 按 §1 保留其一并删掉方括号；`[[If share_hier < 50%: …]]`（附录 B.3）按 kNN-BGE share_hier 处理后删除。

| 稿件占位符 | 值 | 来源 |
| --- | --- | --- |
| `[[ΔMF1 vs kNN-BGE]]` | 7.24 | 0.9892 − kNN-BGE Macro-F1（摘要、4.1、B.3） |
| `[[kNN-BGE MF1]]` | 0.9168 | result_B1_bge.json |
| `[[kNN-BGE WF1]]` | 0.9373 | result_B1_bge.json |
| `[[kNN-E5 MF1]]` | 0.8987 | result_B1_e5.json |
| `[[kNN-E5 WF1]]` | 0.9245 | result_B1_e5.json |
| `[[R5 Acc/MF1]]` | 0.9610/0.9421 | RULE_BASELINE_NEIGHBOR.json |
| `[[R6 Acc/MF1]]` | 0.9917/0.9883 | 同上 |
| `[[R6 coverage]]` | 97.53 | 同上 |
| `[[R5 single-axis Acc]]` | 0.9989 | 同上 |
| `[[R5 multiaxial Acc]]` | 0.9143 | 同上 |
| `[[R6 single-axis Acc]]` | 0.9992 | 同上 |
| `[[R6 multiaxial Acc]]` | 0.9824 | 同上 |
| `[[kNN-BGE share_hier]]` | 76.89 | RULE_BASELINE_NEIGHBOR.json --knn-pred |
| `[[kNN-BGE share_child]]` | 20.02 | 同上 |
| `[[kNN-BGE share_sib]]` | 31.33 | 同上 |
| `[[kNN-BGE share_parent]]` | 25.54 | 同上 |
| `[[kNN-BGE acc_hier]]` | 0.9979 | 同上 |
| `[[kNN-BGE acc_other]]` | 0.7352 | 同上 |
| `[[kNN-BGE single-axis]]` | 0.9422 | 同上 |
| `[[kNN-BGE multiaxial]]` | 0.9311 | 同上 |
| `[[V1 Acc/MF1]]` | 0.9940 ± 0.0006 (n=5) / 0.9907 ± 0.0011 (n=5) | 多种子（n=5） |
| `[[V2 Acc/MF1]]` | 0.9939 ± 0.0009 (n=5) / 0.9909 ± 0.0010 (n=5) | 多种子 |
| `[[C7 Acc/MF1]]` | 0.9854 ± 0.0019 (n=5) / 0.9769 ± 0.0029 (n=5) | 多种子 |
| `[[V1 single-axis Acc]]` | 0.9981 ± 0.0001 (n=5) | MULTIAXIAL_SUMMARY.json |
| `[[V1 multiaxial Acc]]` | 0.9889 ± 0.0014 (n=5) | 同上 |
| `[[V1 multiaxial MF1]]` | 0.9813 ± 0.0028 (n=5) | 同上 |
| `[[E1B ΔPT MF1]]` | +0.00 | §3 SOC 控制，PT Macro-F1，pp |
| `[[SOCDROP ΔPT MF1]]` | -0.10 | 同上 |
| `[[Δ overall]]` | 0.23 | V1 均值 − R6 |
| `[[Δ multiaxial]]` | 0.64 | V1 多轴均值 − R6 多轴 |
| `[[I5 Acc/MF1]]` | 0.8172/0.7167 | INDUCTIVE_RULE_BASELINE.json（仅情况 B 需要） |
| `[[V1 Acc]]` | 0.9940 ± 0.0006 (n=5) | 附录表 A1 |
| `[[V1 MF1]]` | 0.9907 ± 0.0011 (n=5) | 附录表 A1 |
| `[[V1 HLGT F1]]` | 0.8874 ± 0.0202 (n=5) | 附录表 A1 |
| `[[V1 HLT F1]]` | 0.8995 ± 0.0030 (n=5) | 附录表 A1 |
| `[[V1 PT F1]]` | 0.9772 ± 0.0029 (n=5) | 附录表 A1 |
| `[[V1 ΔMF1]]` | — | 附录表 A1 |
| `[[V2 Acc]]` | 0.9939 ± 0.0009 (n=5) | 附录表 A1 |
| `[[V2 MF1]]` | 0.9909 ± 0.0010 (n=5) | 附录表 A1 |
| `[[V2 HLGT F1]]` | 0.8964 ± 0.0247 (n=5) | 附录表 A1 |
| `[[V2 HLT F1]]` | 0.9038 ± 0.0162 (n=5) | 附录表 A1 |
| `[[V2 PT F1]]` | 0.9781 ± 0.0029 (n=5) | 附录表 A1 |
| `[[V2 ΔMF1]]` | +0.02（在 0.93 pp 阈值内；在 V1 ±2 sd 内） | 附录表 A1 |
| `[[C7 Acc]]` | 0.9854 ± 0.0019 (n=5) | 附录表 A1 |
| `[[C7 MF1]]` | 0.9769 ± 0.0029 (n=5) | 附录表 A1 |
| `[[C7 HLGT F1]]` | 0.8686 ± 0.0509 (n=5) | 附录表 A1 |
| `[[C7 HLT F1]]` | 0.9073 ± 0.0126 (n=5) | 附录表 A1 |
| `[[C7 PT F1]]` | 0.9548 ± 0.0053 (n=5) | 附录表 A1 |
| `[[C7 ΔMF1]]` | -1.38（超过 0.93 pp 阈值；超出 V1 ±2 sd） | 附录表 A1 |
| `[[E1B Acc]]` | 0.9935 ± 0.0003 (n=3) | 附录表 A1 |
| `[[E1B MF1]]` | 0.9904 ± 0.0007 (n=3) | 附录表 A1 |
| `[[E1B HLGT F1]]` | 0.8729 ± 0.0493 (n=3) | 附录表 A1 |
| `[[E1B HLT F1]]` | 0.9151 ± 0.0052 (n=3) | 附录表 A1 |
| `[[E1B PT F1]]` | 0.9773 ± 0.0010 (n=3) | 附录表 A1 |
| `[[E1B ΔMF1]]` | -0.03（在 0.93 pp 阈值内；在 V1 ±2 sd 内） | 附录表 A1 |
| `[[SOCDROP Acc]]` | 0.9933 ± 0.0007 (n=3) | 附录表 A1 |
| `[[SOCDROP MF1]]` | 0.9901 ± 0.0006 (n=3) | 附录表 A1 |
| `[[SOCDROP HLGT F1]]` | 0.8324 ± 0.0157 (n=3) | 附录表 A1 |
| `[[SOCDROP HLT F1]]` | 0.9176 ± 0.0187 (n=3) | 附录表 A1 |
| `[[SOCDROP PT F1]]` | 0.9762 ± 0.0029 (n=3) | 附录表 A1 |
| `[[SOCDROP ΔMF1]]` | -0.06（在 0.93 pp 阈值内；在 V1 ±2 sd 内） | 附录表 A1 |
| `[[C3 Acc]]` | 0.9903 (n=1) | 附录表 A1（SKIP 组，不跑：稿件删除 C3 行，可选） |
| `[[C3 MF1]]` | 0.9846 (n=1) | 附录表 A1（SKIP 组，不跑：稿件删除 C3 行，可选） |
| `[[C3 HLGT F1]]` | 0.8784 (n=1) | 附录表 A1（SKIP 组，不跑：稿件删除 C3 行，可选） |
| `[[C3 HLT F1]]` | 0.9220 (n=1) | 附录表 A1（SKIP 组，不跑：稿件删除 C3 行，可选） |
| `[[C3 PT F1]]` | 0.9636 (n=1) | 附录表 A1（SKIP 组，不跑：稿件删除 C3 行，可选） |
| `[[C3 ΔMF1]]` | -0.60（在 0.93 pp 阈值内；超出 V1 ±2 sd） | 附录表 A1（SKIP 组，不跑：稿件删除 C3 行，可选） |
| `[[V3 Acc]]` | 0.9807 ± 0.0056 (n=5) | 附录表 A1 |
| `[[V3 MF1]]` | 0.9692 ± 0.0094 (n=5) | 附录表 A1 |
| `[[V3 HLGT F1]]` | 0.8675 ± 0.0000 (n=5) | 附录表 A1 |
| `[[V3 HLT F1]]` | 0.8746 ± 0.0100 (n=5) | 附录表 A1 |
| `[[V3 PT F1]]` | 0.9402 ± 0.0133 (n=5) | 附录表 A1 |
| `[[V3 ΔMF1]]` | -2.15（超过 0.93 pp 阈值；超出 V1 ±2 sd） | 附录表 A1 |
| `[[V4 Acc]]` | 0.9883 ± 0.0027 (n=5) | 附录表 A1 |
| `[[V4 MF1]]` | 0.9817 ± 0.0042 (n=5) | 附录表 A1 |
| `[[V4 HLGT F1]]` | 0.8675 ± 0.0000 (n=5) | 附录表 A1 |
| `[[V4 HLT F1]]` | 0.8974 ± 0.0073 (n=5) | 附录表 A1 |
| `[[V4 PT F1]]` | 0.9559 ± 0.0078 (n=5) | 附录表 A1 |
| `[[V4 ΔMF1]]` | -0.89（在 0.93 pp 阈值内；超出 V1 ±2 sd） | 附录表 A1 |
| `[[E1A Acc]]` | 0.9942 ± 0.0013 (n=3) | 附录表 A1 |
| `[[E1A MF1]]` | 0.9914 ± 0.0016 (n=3) | 附录表 A1 |
| `[[E1A HLGT F1]]` | 0.8784 ± 0.0000 (n=3) | 附录表 A1 |
| `[[E1A HLT F1]]` | 0.9098 ± 0.0081 (n=3) | 附录表 A1 |
| `[[E1A PT F1]]` | 0.9803 ± 0.0010 (n=3) | 附录表 A1 |
| `[[E1A ΔMF1]]` | +0.07（在 0.93 pp 阈值内；在 V1 ±2 sd 内） | 附录表 A1 |

## 4c. 归纳式（disjoint 模型选择）与父×子联合诊断（多种子）

| 数字 | 值 |
| --- | --- |
| IND-SAGE-KEEP disjoint Acc | 0.9280 ± 0.0013 (n=5) |
| IND-SAGE-KEEP disjoint MF1 | 0.8512 ± 0.0052 (n=5) |
| IND-NOGNN-PUBMEDBERT disjoint Acc | 0.7445 (n=1) |
| IND-NOGNN-PUBMEDBERT disjoint MF1 | 0.6339 (n=1) |
| IND-SAGE-KEEP 选择乐观偏差 (pp, 第一轮 − disjoint, seed 42) | +0.35 |
| IND-SAGE-MASK 选择乐观偏差 (pp, seed 42) | +12.59 |
| IND-NOGNN-PUBMEDBERT 选择乐观偏差 (pp, seed 42) | +0.08 |
| KEEP − NoGNN (disjoint, pp) | +18.34 |
| MASK − NoGNN (disjoint, pp) | -32.18 |
| KEEP − I5 (disjoint, pp) | +11.08 |
| V1 联合差 (pp) | +2.11 ± 0.38 (n=5) |
| V1 父链分量 (pp) | -0.26 ± 0.34 (n=5) |
| V1 子链分量 (pp) | +2.37 ± 0.59 (n=5) |
| V1 一维父链净效应 (pp) | +1.19 ± 0.20 (n=5) |
| V4 − V3 Macro-F1 (pp) | +1.25（V3 n=5，V4 n=5，合并 sd = 1.03 pp） |
| V4 − V3 Acc (pp) | +0.76 |
| '补偿作用'一句 | 不成立：V4 − V3 未同时超过 0.93 pp 与 2 倍合并 sd，稿件需改为'未观察到稳定的补偿作用' |

## 5. 缺失输入 / 需要注意

- 无


## 6. SUBMIT 组完整性与结论判定

> 2026-10-29 17:20 起只有 SUBMIT / SKIP 两组，SUBMIT 一次跑完，这里不再判定"要不要再跑"，只给出各结论句的写法。

| 事项 | 判定 |
| --- | --- |
| 近循环（SOC 控制） | 三种控制在 HLT / PT 上都没有超出 V1 ±2 sd 的下降：SOC 名称、身份与结构都不是分类捷径；E1A 的近零效应与设计预期一致 |
| E0 结论句 | 情况 B：V1 − R6 = +0.23 pp（未达到 ≥ 1 pp 且超出 V1 ±2 sd 的标准，2 sd = 0.11 pp） |
| 表 5 口径 | 最大乐观偏差 12.59 pp > 0.58 pp：表 5 的 SAGE / NoGNN 行改用 disjoint 结果，第一轮值移到附录；GAT 归纳式两行未用新协议重跑，移出表 5 或注明第一轮协议 |
| 归纳式泛化证据（KEEP − I5） | +11.08 pp（KEEP n=5，2 sd = 0.27 pp）→ 成立：情况 B 的泛化论证可以写 |
| '补偿作用'一句（4.2 / 5.1） | 改写为'未观察到稳定的补偿作用'（V4 − V3 = +1.25 pp） |
