# 单轴 / 多轴测试节点上的表现（离线复算）

> 单轴 = 该节点在长表中只落在 1 个 SOC 下；多轴 = n_SOC ≥ 2。只评估主 SOC。
> 多个种子给 mean ± sd（ddof=1）。`random` = 在每个多轴节点的候选 SOC 中均匀随机选一个的期望准确率。

## V1（5 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9940 ± 0.0006 (n=5) | 0.9907 ± 0.0011 (n=5) |
| 单轴 | 6638 | 0.9981 ± 0.0001 (n=5) | 0.9956 ± 0.0003 (n=5) |
| 多轴 | 5403 | 0.9889 ± 0.0014 (n=5) | 0.9813 ± 0.0028 (n=5) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=5) | — |
| 多轴 · HLT | 14 | 1.0000 ± 0.0000 (n=5) | — |
| 多轴 · PT | 1162 | 0.9690 ± 0.0033 (n=5) | — |
| 多轴 · LLT | 4226 | 0.9943 ± 0.0011 (n=5) | — |

- 错误数：72.6 ± 6.7 (n=5)；其中多轴节点：60.2 ± 7.5 (n=5)；占比：82.73% ± 2.61 (n=5)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/V1_S43/test_predictions_SAGE.json`：all 0.9947 / single 0.9980 / multi 0.9906（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V1_S44/test_predictions_SAGE.json`：all 0.9939 / single 0.9982 / multi 0.9887（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V1_S45/test_predictions_SAGE.json`：all 0.9943 / single 0.9980 / multi 0.9896（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V1_S46/test_predictions_SAGE.json`：all 0.9938 / single 0.9980 / multi 0.9885（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/analysis_out/exported_predictions/test_predictions_SAGE__V1.json`：all 0.9932 / single 0.9983 / multi 0.9869（n=12041，未匹配 0）

## V2（4 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9939 ± 0.0010 (n=4) | 0.9908 ± 0.0012 (n=4) |
| 单轴 | 6638 | 0.9982 ± 0.0002 (n=4) | 0.9957 ± 0.0004 (n=4) |
| 多轴 | 5403 | 0.9886 ± 0.0024 (n=4) | 0.9810 ± 0.0037 (n=4) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=4) | — |
| 多轴 · HLT | 14 | 1.0000 ± 0.0000 (n=4) | — |
| 多轴 · PT | 1162 | 0.9686 ± 0.0050 (n=4) | — |
| 多轴 · LLT | 4226 | 0.9940 ± 0.0019 (n=4) | — |

- 错误数：74.0 ± 12.2 (n=4)；其中多轴节点：61.8 ± 13.2 (n=4)；占比：82.99% ± 3.92 (n=4)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/V2_S43/test_predictions_SAGE.json`：all 0.9942 / single 0.9982 / multi 0.9893（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V2_S44/test_predictions_SAGE.json`：all 0.9946 / single 0.9979 / multi 0.9906（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V2_S45/test_predictions_SAGE.json`：all 0.9943 / single 0.9982 / multi 0.9895（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V2_S46/test_predictions_SAGE.json`：all 0.9924 / single 0.9983 / multi 0.9850（n=12041，未匹配 0）

## C7（3 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9858 ± 0.0016 (n=3) | 0.9772 ± 0.0023 (n=3) |
| 单轴 | 6638 | 0.9978 ± 0.0002 (n=3) | 0.9953 ± 0.0008 (n=3) |
| 多轴 | 5403 | 0.9711 ± 0.0035 (n=3) | 0.9451 ± 0.0077 (n=3) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=3) | — |
| 多轴 · HLT | 14 | 0.9762 ± 0.0412 (n=3) | — |
| 多轴 · PT | 1162 | 0.9429 ± 0.0059 (n=3) | — |
| 多轴 · LLT | 4226 | 0.9789 ± 0.0030 (n=3) | — |

- 错误数：170.7 ± 19.6 (n=3)；其中多轴节点：156.0 ± 18.7 (n=3)；占比：91.37% ± 0.64 (n=3)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/C7_S44/test_predictions_SAGE.json`：all 0.9875 / single 0.9979 / multi 0.9748（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/C7_S45/test_predictions_SAGE.json`：all 0.9843 / single 0.9976 / multi 0.9680（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/C7_S46/test_predictions_SAGE.json`：all 0.9856 / single 0.9979 / multi 0.9706（n=12041，未匹配 0）

## V3（4 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9822 ± 0.0052 (n=4) | 0.9720 ± 0.0080 (n=4) |
| 单轴 | 6638 | 0.9970 ± 0.0003 (n=4) | 0.9935 ± 0.0007 (n=4) |
| 多轴 | 5403 | 0.9640 ± 0.0115 (n=4) | 0.9344 ± 0.0188 (n=4) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=4) | — |
| 多轴 · HLT | 14 | 0.9821 ± 0.0357 (n=4) | — |
| 多轴 · PT | 1162 | 0.9318 ± 0.0195 (n=4) | — |
| 多轴 · LLT | 4226 | 0.9727 ± 0.0096 (n=4) | — |

- 错误数：214.5 ± 62.3 (n=4)；其中多轴节点：194.8 ± 62.2 (n=4)；占比：90.20% ± 2.93 (n=4)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/V3_S43/test_predictions_SAGE.json`：all 0.9762 / single 0.9968 / multi 0.9510（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V3_S44/test_predictions_SAGE.json`：all 0.9860 / single 0.9967 / multi 0.9728（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V3_S45/test_predictions_SAGE.json`：all 0.9795 / single 0.9973 / multi 0.9576（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V3_S46/test_predictions_SAGE.json`：all 0.9870 / single 0.9973 / multi 0.9745（n=12041，未匹配 0）

## V4（4 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9882 ± 0.0031 (n=4) | 0.9817 ± 0.0048 (n=4) |
| 单轴 | 6638 | 0.9974 ± 0.0001 (n=4) | 0.9943 ± 0.0004 (n=4) |
| 多轴 | 5403 | 0.9768 ± 0.0068 (n=4) | 0.9591 ± 0.0128 (n=4) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=4) | — |
| 多轴 · HLT | 14 | 0.9821 ± 0.0357 (n=4) | — |
| 多轴 · PT | 1162 | 0.9486 ± 0.0118 (n=4) | — |
| 多轴 · LLT | 4226 | 0.9846 ± 0.0055 (n=4) | — |

- 错误数：142.5 ± 37.3 (n=4)；其中多轴节点：125.2 ± 37.0 (n=4)；占比：87.32% ± 3.16 (n=4)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/V4_S43/test_predictions_SAGE.json`：all 0.9876 / single 0.9974 / multi 0.9756（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V4_S44/test_predictions_SAGE.json`：all 0.9911 / single 0.9973 / multi 0.9835（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V4_S45/test_predictions_SAGE.json`：all 0.9841 / single 0.9973 / multi 0.9678（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/V4_S46/test_predictions_SAGE.json`：all 0.9899 / single 0.9976 / multi 0.9804（n=12041，未匹配 0）

## E1A（3 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9942 ± 0.0013 (n=3) | 0.9914 ± 0.0016 (n=3) |
| 单轴 | 6638 | 0.9982 ± 0.0002 (n=3) | 0.9960 ± 0.0005 (n=3) |
| 多轴 | 5403 | 0.9893 ± 0.0027 (n=3) | 0.9827 ± 0.0039 (n=3) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=3) | — |
| 多轴 · HLT | 14 | 1.0000 ± 0.0000 (n=3) | — |
| 多轴 · PT | 1162 | 0.9722 ± 0.0028 (n=3) | — |
| 多轴 · LLT | 4226 | 0.9940 ± 0.0026 (n=3) | — |

- 错误数：69.3 ± 15.7 (n=3)；其中多轴节点：57.7 ± 14.4 (n=3)；占比：82.92% ± 1.95 (n=3)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/E1A_S42/test_predictions_SAGE.json`：all 0.9928 / single 0.9980 / multi 0.9863（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/E1A_S43/test_predictions_SAGE.json`：all 0.9953 / single 0.9985 / multi 0.9913（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/E1A_S44/test_predictions_SAGE.json`：all 0.9947 / single 0.9982 / multi 0.9904（n=12041，未匹配 0）

## E1B（3 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9935 ± 0.0003 (n=3) | 0.9904 ± 0.0007 (n=3) |
| 单轴 | 6638 | 0.9982 ± 0.0000 (n=3) | 0.9960 ± 0.0003 (n=3) |
| 多轴 | 5403 | 0.9877 ± 0.0007 (n=3) | 0.9802 ± 0.0019 (n=3) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=3) | — |
| 多轴 · HLT | 14 | 1.0000 ± 0.0000 (n=3) | — |
| 多轴 · PT | 1162 | 0.9679 ± 0.0013 (n=3) | — |
| 多轴 · LLT | 4226 | 0.9931 ± 0.0008 (n=3) | — |

- 错误数：78.3 ± 4.0 (n=3)；其中多轴节点：66.3 ± 4.0 (n=3)；占比：84.65% ± 0.80 (n=3)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/E1B_S42/test_predictions_SAGE.json`：all 0.9934 / single 0.9982 / multi 0.9876（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/E1B_S43/test_predictions_SAGE.json`：all 0.9932 / single 0.9982 / multi 0.9870（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/E1B_S44/test_predictions_SAGE.json`：all 0.9939 / single 0.9982 / multi 0.9885（n=12041，未匹配 0）

## SOCDROP（3 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9933 ± 0.0007 (n=3) | 0.9901 ± 0.0006 (n=3) |
| 单轴 | 6638 | 0.9980 ± 0.0002 (n=3) | 0.9956 ± 0.0005 (n=3) |
| 多轴 | 5403 | 0.9876 ± 0.0018 (n=3) | 0.9790 ± 0.0018 (n=3) |
| 多轴 · HLGT | 1 | 1.0000 ± 0.0000 (n=3) | — |
| 多轴 · HLT | 14 | 0.9762 ± 0.0412 (n=3) | — |
| 多轴 · PT | 1162 | 0.9673 ± 0.0052 (n=3) | — |
| 多轴 · LLT | 4226 | 0.9932 ± 0.0011 (n=3) | — |

- 错误数：80.3 ± 8.7 (n=3)；其中多轴节点：67.0 ± 9.8 (n=3)；占比：83.18% ± 3.08 (n=3)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/data/jbi_runs_review/SOCDROP_S42/test_predictions_SAGE.json`：all 0.9925 / single 0.9982 / multi 0.9856（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/SOCDROP_S43/test_predictions_SAGE.json`：all 0.9939 / single 0.9979 / multi 0.9891（n=12041，未匹配 0）
  - `/mnt/workspace/JBI/data/jbi_runs_review/SOCDROP_S44/test_predictions_SAGE.json`：all 0.9935 / single 0.9979 / multi 0.9882（n=12041，未匹配 0）

## kNN-BGE（1 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9372 (n=1) | 0.9168 (n=1) |
| 单轴 | 6638 | 0.9422 (n=1) | 0.9045 (n=1) |
| 多轴 | 5403 | 0.9311 (n=1) | 0.7925 (n=1) |
| 多轴 · HLGT | 1 | 1.0000 (n=1) | — |
| 多轴 · HLT | 14 | 0.9286 (n=1) | — |
| 多轴 · PT | 1162 | 0.9690 (n=1) | — |
| 多轴 · LLT | 4226 | 0.9207 (n=1) | — |

- 错误数：756；其中多轴节点：372；占比：49.21% (n=1)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/BaseLine/baseline_outputs/kNN_bge/test_predictions_kNN_bge.json`：all 0.9372 / single 0.9422 / multi 0.9311（n=12041，未匹配 0）

## kNN-E5（1 个 run）

| 子集 | n | Acc | Macro-F1 |
| --- | --- | --- | --- |
| 全部 | 12041 | 0.9244 (n=1) | 0.8987 (n=1) |
| 单轴 | 6638 | 0.9318 (n=1) | 0.8865 (n=1) |
| 多轴 | 5403 | 0.9154 (n=1) | 0.7756 (n=1) |
| 多轴 · HLGT | 1 | 1.0000 (n=1) | — |
| 多轴 · HLT | 14 | 0.8571 (n=1) | — |
| 多轴 · PT | 1162 | 0.9561 (n=1) | — |
| 多轴 · LLT | 4226 | 0.9044 (n=1) | — |

- 错误数：910；其中多轴节点：457；占比：50.22% (n=1)
- 多轴节点上随机选候选 SOC 的期望准确率：0.4616

  - `/mnt/workspace/JBI/BaseLine/baseline_outputs/kNN_e5/test_predictions_kNN_e5.json`：all 0.9244 / single 0.9318 / multi 0.9154（n=12041，未匹配 0）

## 汇总（便于直接写入正文）

| 组 | 单轴 Acc | 多轴 Acc | 多轴 Macro-F1 | 错误中多轴占比 |
| --- | --- | --- | --- | --- |
| V1 | 0.9981 ± 0.0001 (n=5) | 0.9889 ± 0.0014 (n=5) | 0.9813 ± 0.0028 (n=5) | 82.73% ± 2.61 (n=5) |
| V2 | 0.9982 ± 0.0002 (n=4) | 0.9886 ± 0.0024 (n=4) | 0.9810 ± 0.0037 (n=4) | 82.99% ± 3.92 (n=4) |
| C7 | 0.9978 ± 0.0002 (n=3) | 0.9711 ± 0.0035 (n=3) | 0.9451 ± 0.0077 (n=3) | 91.37% ± 0.64 (n=3) |
| V3 | 0.9970 ± 0.0003 (n=4) | 0.9640 ± 0.0115 (n=4) | 0.9344 ± 0.0188 (n=4) | 90.20% ± 2.93 (n=4) |
| V4 | 0.9974 ± 0.0001 (n=4) | 0.9768 ± 0.0068 (n=4) | 0.9591 ± 0.0128 (n=4) | 87.32% ± 3.16 (n=4) |
| E1A | 0.9982 ± 0.0002 (n=3) | 0.9893 ± 0.0027 (n=3) | 0.9827 ± 0.0039 (n=3) | 82.92% ± 1.95 (n=3) |
| E1B | 0.9982 ± 0.0000 (n=3) | 0.9877 ± 0.0007 (n=3) | 0.9802 ± 0.0019 (n=3) | 84.65% ± 0.80 (n=3) |
| SOCDROP | 0.9980 ± 0.0002 (n=3) | 0.9876 ± 0.0018 (n=3) | 0.9790 ± 0.0018 (n=3) | 83.18% ± 3.08 (n=3) |
| kNN-BGE | 0.9422 (n=1) | 0.9311 (n=1) | 0.7925 (n=1) | 49.21% (n=1) |
| kNN-E5 | 0.9318 (n=1) | 0.9154 (n=1) | 0.7756 (n=1) | 50.22% (n=1) |
