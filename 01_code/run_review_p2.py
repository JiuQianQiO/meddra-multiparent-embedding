# -*- coding: utf-8 -*-
"""run_review_p2.py —— JBI 审稿补实验（P2）调度器：只生成命令，不实际执行。

三类任务：
  1. 归纳式多种子（IND-SAGE-MASK / IND-NOGNN-PUBMEDBERT）—— 补跑种子 43–46（42 已在表 5）；
  2. B2-ComplEx 塌缩诊断 —— 重跑（旧脚本有"逐 epoch 调用致损失不变"的 bug，已修）；
  3. LP/C&S 有监督图基线 —— 重跑（旧脚本有"SOC 零标签污染"的 bug，已修）。

【关键】归纳式训练器在"带实验名参数"（python X.py IND-SAGE-MASK）时走启动器模式，
      其输出目录 = $MEDDRA_OUTPUT_ROOT/实验名（不含 seed），多种子会互相覆盖。
      因此这里改用"主体模式"（不带实验名参数）：用 MEDDRA_RUN_DIR 直接指定含 seed 的目录，
      MEDDRA_EXPERIMENT 指定实验名（解析注册表里的 keep_edges 配置）。

默认路径与服务器初始化脚本 jbi_env_290.sh 完全一致。

用法（本地生成命令，复制到服务器执行）：
  python run_review_p2.py --print-commands
  python run_review_p2.py --print-commands --groups inductive   # 只看归纳式
  python run_review_p2.py --print-commands --seeds 43 44 45 46  # 默认 43–46
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-commands", action="store_true", help="打印可复制的 bash 命令")
    ap.add_argument("--groups", default="all", help="all / inductive / b2 / lpcs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[43, 44],
                    help="训练种子（默认 43 44；加上已有 42 共 n=3，与 GAT/NoGNN 参照口径一致）")
    ap.add_argument("--script-dir", default="/mnt/workspace/JBI/review_p2",
                    help="服务器上脚本所在目录")
    ap.add_argument("--csv", default="/mnt/workspace/Data/MEDDRA_DATA_29_0.csv")
    ap.add_argument("--cache-dir", default="/mnt/workspace/JBI/data/cache_shared")
    ap.add_argument("--text-model-dir", default="/data/hf_models/pubmedbert")
    ap.add_argument("--split-index", default="/mnt/workspace/JBI/data/jbi_runs/split_indices.json")
    ap.add_argument("--run-root", default="/mnt/workspace/JBI/data/jbi_runs")
    args = ap.parse_args()

    cd = "cd %s" % args.script_dir
    cmds = []

    def env(extra_pairs):
        base = [
            ("MEDDRA_CSV_PATH", args.csv),
            ("MEDDRA_CACHE_DIR", args.cache_dir),
            ("MEDDRA_TEXT_MODEL_DIR", args.text_model_dir),
            ("MEDDRA_SPLIT_INDEX_FILE", args.split_index),
            ("MEDDRA_SPLIT_SEED", "42"),
            ("PYTHONIOENCODING", "utf-8"),
        ]
        return " ".join("%s=%s" % (k, v) for k, v in base + extra_pairs)

    # 1) 归纳式多种子（主体模式：不带实验名参数）
    for seed in args.seeds:
        e = env([
            ("MEDDRA_RUN_DIR", "%s/IND-SAGE-MASK/seed%d" % (args.run_root, seed)),
            ("MEDDRA_EXPERIMENT", "IND-SAGE-MASK"),
            ("MEDDRA_TRAIN_SEED", str(seed)),
            ("MEDDRA_IND_SELECT", "disjoint"),
        ])
        cmds.append(("IND-SAGE-MASK/seed%d" % seed, "补多种子（42 已跑）",
                     "%s && %s python Meddra_inductive_SAGE.py" % (cd, e)))
    for seed in args.seeds:
        e = env([
            ("MEDDRA_RUN_DIR", "%s/IND-NOGNN-PUBMEDBERT/seed%d" % (args.run_root, seed)),
            ("MEDDRA_EXPERIMENT", "IND-NOGNN-PUBMEDBERT"),
            ("MEDDRA_TRAIN_SEED", str(seed)),
            ("MEDDRA_IND_SELECT", "disjoint"),
        ])
        cmds.append(("IND-NOGNN-PUBMEDBERT/seed%d" % seed, "补多种子（42 已跑）",
                     "%s && %s python Meddra_inductive_noGNN.py" % (cd, e)))

    # 2) B2-ComplEx 诊断（重跑）
    cmds.append(("B2-COMPLEX-DIAG", "重跑（修复损失曲线 bug）",
                 "%s && python baseline/diagnose_b2_complex.py --csv %s --cache-dir %s --out-dir %s/B2 --epochs 100 --dim 256 --seed 42"
                 % (cd, args.csv, args.cache_dir, args.run_root)))

    # 3) LP/C&S 基线（重跑）
    cmds.append(("LP-CS", "重跑（修复 SOC 零标签污染 bug）",
                 "%s && python baseline/lp_cs_baseline.py --csv %s --cache-dir %s --out-dir %s/LP_CS --lp-iters 50 --cs-alpha 0.8 --cs-iters 20"
                 % (cd, args.csv, args.cache_dir, args.run_root)))

    grp = args.groups
    filtered = []
    for name, note, cmd in cmds:
        if grp == "all":
            filtered.append((name, note, cmd))
        elif grp == "inductive" and "IND-" in name:
            filtered.append((name, note, cmd))
        elif grp == "b2" and "B2" in name:
            filtered.append((name, note, cmd))
        elif grp == "lpcs" and "LP-CS" in name:
            filtered.append((name, note, cmd))

    print("#" * 78)
    print("# JBI 审稿补实验（P2）—— %d 条命令（组=%s，种子=%s）" % (len(filtered), grp, args.seeds))
    print("# 用法：先 source jbi_env_290.sh，再逐条复制执行")
    print("#" * 78)
    for name, note, cmd in filtered:
        print("\n# ---- %s （%s）----" % (name, note))
        print(cmd)
    print("\n# 完成后用 compile_inductive_seeds.py 汇总多种子 mean±sd。")


if __name__ == "__main__":
    main()
