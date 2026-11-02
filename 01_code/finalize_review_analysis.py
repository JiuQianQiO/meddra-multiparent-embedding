# -*- coding: utf-8 -*-
r"""finalize_review_analysis.py — 多种子训练完成后，一次生成汇总与子群表。

不是训练，仅调用离线脚本：
  1) compile_review_seed_metrics.py → REVIEW_SEED_metrics.md/csv
  2) compile_jbi_metrics.py         → REVIEW_metrics.md
  3) subgroup_performance.py        → REVIEW_subgroups.md
  4) subgroup_multiaxial.py         → REVIEW_multiaxial.md（E4 多父/多SOC统计）
  5) diagnose_joint_subgroup.py     → 父×子二维联合诊断（可选，--joint-dirs 指定 run 目录）

用法：
  python finalize_review_analysis.py \
    --review-root /mnt/workspace/JBI/data/jbi_runs_review \
    --csv         /mnt/workspace/Data/MEDDRA_DATA_29_0.csv \
    --split       /mnt/workspace/JBI/data/jbi_runs/split_indices.json \
    --out-dir     /mnt/workspace/JBI/analysis_out \
    --base-root   /mnt/workspace/JBI/data/jbi_runs \
    --joint-dirs  /mnt/workspace/JBI/data/jbi_runs_review/V1_S43 \
                  /mnt/workspace/JBI/data/jbi_runs/SAGE/C1V1 \
                  /mnt/workspace/JBI/data/jbi_runs/GAT
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args):
    cmd = [sys.executable, os.path.join(HERE, script)] + args
    print(">>>", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print("[fail] rc=%d  %s" % (rc, script), flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--base-root", default="")
    ap.add_argument("--joint-dirs", nargs="*", default=[],
                    help="父×子二维联合诊断（diagnose_joint_subgroup.py）要诊断的 run 目录（含 best_checkpoint_*.pt）；留空则跳过")
    ap.add_argument("--subgroup-only", default="",
                    help="传给 subgroup_performance.py 的 --only（逗号分隔的 run 目录名）；留空则扫全部")
    a = ap.parse_args()
    out = a.out_dir or a.review_root
    os.makedirs(out, exist_ok=True)

    base = a.base_root
    if not base and a.review_root.endswith("_review"):
        base = a.review_root[:-len("_review")]

    seed_metrics_md = os.path.join(out, "REVIEW_SEED_metrics.md")
    seed_metrics_csv = os.path.join(out, "REVIEW_SEED_metrics.csv")
    metrics_md = os.path.join(out, "REVIEW_metrics.md")
    subgroup_md = os.path.join(out, "REVIEW_subgroups.md")

    seed_args = ["--root", a.review_root]
    if base:
        seed_args += ["--base-root", base]
    seed_args += ["--md", seed_metrics_md, "--csv", seed_metrics_csv]

    run("compile_review_seed_metrics.py", seed_args)
    run("compile_jbi_metrics.py", ["--root", a.review_root, "--md", metrics_md])
    subgroup_args = ["--root", a.review_root, "--csv", a.csv,
                     "--split", a.split, "--md", subgroup_md]
    if a.subgroup_only:
        subgroup_args += ["--only", a.subgroup_only]
    run("subgroup_performance.py", subgroup_args)
    run("subgroup_multiaxial.py", ["--csv", a.csv, "--split", a.split,
                                   "--md", os.path.join(out, "REVIEW_multiaxial.md")])

    if a.joint_dirs:
        run("diagnose_joint_subgroup.py",
            ["--run-dirs"] + list(a.joint_dirs)
            + ["--split-index", a.split, "--out-dir", out])
        print("父×子二维联合诊断结果（joint_subgroup_diag.json）写入各 run 目录 / --out-dir=%s" % out, flush=True)
    else:
        print("跳过父×子二维联合诊断：未提供 --joint-dirs。"
              "例如 --joint-dirs <review>/V1_S43 <review>/C7_S44 <base>/SAGE/C1V1 <base>/GAT "
              "（各目录需含 best_checkpoint_*.pt）", flush=True)

    print("done:", seed_metrics_md, metrics_md, subgroup_md,
          os.path.join(out, "REVIEW_multiaxial.md"), flush=True)


if __name__ == "__main__":
    main()
