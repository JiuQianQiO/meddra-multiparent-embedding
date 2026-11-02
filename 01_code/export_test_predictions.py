# -*- coding: utf-8 -*-
r"""export_test_predictions.py — 用 checkpoint 做一次前向，导出独立测试集的逐节点预测。

用途：
  审稿意见 M3 要求量化「测试 PT 的同名 LLT 是否在训练集」这一潜在泄漏通道。
  官方产物目录没有逐节点预测文件（只有 test_metrics_*.json 汇总），本脚本不重训、
  只读 checkpoint + 图/特征缓存 + split_indices，重建模型后做一次前向，导出
  test_predictions_SAGE.json（node_code / level / true_soc_code / pred_soc_code / correct）。

  回传后在本地用 split_indices.json + MEDDRA_DATA_29_0.csv 即可算「孪生 LLT 在/不在
  训练集」两组准确率（见 ANALYSIS.md 的分组脚本）。

用法（服务器，实验目录或任意目录）：
  python export_test_predictions.py \
      --trainer   /mnt/workspace/JBI/review/Meddra_full_trainer_SAGE.py \
      --ckpt      /mnt/workspace/JBI/data/jbi_runs/SAGE/C1V1/best_checkpoint_*.pt \
      --cache-dir /mnt/workspace/JBI/data/cache_shared \
      --csv       /mnt/workspace/Data/MEDDRA_DATA_29_0.csv \
      --split     /mnt/workspace/JBI/data/jbi_runs/split_indices.json \
      --out-dir   /mnt/workspace/JBI/analysis_out

说明：
  - 只支持转导式 SAGE/GAT/noGNN（三分协议）的 C1V1 checkpoint；归纳式见 channel_diag.py。
  - 前向约几分钟（CPU 亦可，GPU 更快）；不更新任何权重。
  - 脚本会做自洽校验：复现的 train/val/test 规模必须等于 97,641/10,837/12,041，
    且重算的测试集 Acc 应与 checkpoint 记录一致（相差 >1e-3 即告警），防止重建错误。
"""
import argparse
import glob
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from dataclasses import fields as dc_fields

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXPECT = {"train": 97641, "val": 10837, "test": 12041}


def load_trainer_module(path: str):
    spec = importlib.util.spec_from_file_location("meddra_trainer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_config(mod, cls, raw: dict):
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in (raw or {}).items() if k in valid})


def find_level_tensors(obj, node_types, depth=0):
    if depth > 4 or not isinstance(obj, dict):
        return {}
    hit = {k: v for k, v in obj.items() if k in node_types and hasattr(v, "shape")}
    if hit:
        return hit
    for v in obj.values():
        got = find_level_tensors(v, node_types, depth + 1)
        if got:
            return got
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="导出独立测试集逐节点预测（一次前向，不训练）")
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--csv", default=os.environ.get("MEDDRA_CSV_PATH", ""))
    ap.add_argument("--split", default=os.environ.get("MEDDRA_SPLIT_INDEX_FILE", ""))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(args.ckpt))
    if not ckpts:
        print("找不到 checkpoint:", args.ckpt)
        return 2
    ckpt_path = ckpts[0]
    ck_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cache_dir = args.cache_dir or os.path.join(ck_dir, "meddra_cache")
    if not os.path.isdir(cache_dir) and glob.glob(os.path.join(ck_dir, "*_graph.pt")):
        cache_dir = ck_dir

    # 用 split_indices 保证 test 集与官方一致（trainer 会逐成员校验）
    if args.split:
        os.environ["MEDDRA_SPLIT_INDEX_FILE"] = args.split
    if args.csv:
        os.environ["MEDDRA_CSV_PATH"] = args.csv
    os.environ["MEDDRA_RUN_DIR"] = tempfile.mkdtemp(prefix="exportpred_")

    print("=" * 90)
    print("trainer :", args.trainer)
    print("ckpt    :", ckpt_path)
    print("cache   :", cache_dir, "(存在=%s)" % os.path.isdir(cache_dir))
    print("csv     :", args.csv or "(缺)")
    print("split   :", args.split or "(缺，将由 trainer 重新划分)")
    print("=" * 90)

    mod = load_trainer_module(args.trainer)
    dev = getattr(mod, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model, ck = mod.ModelExporter.load_checkpoint(ckpt_path, device=dev)
    model.eval()
    tc = rebuild_config(mod, mod.TrainingConfig, ck.get("training_config"))
    cc = (rebuild_config(mod, mod.ContrastiveConfig, ck.get("contrastive_config"))
          if hasattr(mod, "ContrastiveConfig") else None)
    dim = ck.get("embedding_dim", 256)
    print("[1/5] 模型已重建，记录 Acc=%.4f, dim=%d" % (
        ck.get("best_metrics", {}).get("accuracy", 0.0), dim))

    csv_path = args.csv or (ck.get("training_config", {}) or {}).get("meddra_csv_path", "")
    if not csv_path:
        print("需要 --csv 或 MEDDRA_CSV_PATH")
        return 2
    cm = mod.GraphCacheManager(cache_dir)
    proc = mod.HeteroMedDRADataProcessor(csv_path, cm)
    graph, _ = proc.load_or_build_graph(embedding_dim=dim, force_rebuild=False, verbose=True)
    if graph is None:
        print("图加载失败（缓存缺失或版本不匹配，需先在服务器重建图+特征缓存）")
        return 2
    print("[2/5] 图已就绪：%d 节点" % len(proc.node_codes))

    _mc = ck.get("model_config", {}) or {}
    feat_type = (_mc.get("feature_type") if isinstance(_mc, dict)
                 else getattr(_mc, "feature_type", None)) or "pubmedbert"
    features, _fdim = mod.get_pubmedbert_features(
        processor=proc, cache_manager=cm, embedding_dim=dim,
        force_rebuild=False, feature_type=feat_type)
    feat = find_level_tensors(features, set(proc.graph_data.node_types))
    if not feat:
        for pat in (f"*_{feat_type}.pt", "*_pubmedbert.pt", "*features*.pt"):
            for p in glob.glob(os.path.join(cache_dir, pat)):
                if "graph" in os.path.basename(p):
                    continue
                try:
                    obj = torch.load(p, map_location="cpu", weights_only=False)
                except Exception:
                    continue
                feat = find_level_tensors(obj, set(proc.graph_data.node_types))
                if feat:
                    break
            if feat:
                break
    exp = {}
    for k, v in model.state_dict().items():
        if k.endswith(".0.weight") and "node_lin" in k:
            exp[k.split("node_lin.")[-1].split(".")[0]] = int(v.shape[1])
    assigned = []
    for ntype, fts in (feat or {}).items():
        if ntype not in proc.graph_data.node_types or not hasattr(fts, "shape"):
            continue
        want = exp.get(ntype)
        if want is not None and int(fts.shape[1]) != want:
            print("⚠️ %s 特征维度 %d != 模型期望 %d → 跳过" % (ntype, int(fts.shape[1]), want))
            continue
        proc.graph_data[ntype].x = fts.float()
        assigned.append(ntype)
    print("[3/5] 特征已装配：%s（feature_type=%s）" % (sorted(assigned), feat_type))

    sig = inspect.signature(mod.HeteroMedDRATrainer.__init__)
    kw = {}
    for opt in ("model_input_config", "data_config"):
        if opt in sig.parameters:
            v = getattr(mod, "ModelInputConfig" if opt == "model_input_config" else "DataConfig", None)
            if v is not None:
                try:
                    kw[opt] = v()
                except Exception:
                    pass
    if "contrastive_config" in sig.parameters and cc is not None:
        kw["contrastive_config"] = cc
    trainer = mod.HeteroMedDRATrainer(model=model, data=proc.graph_data, processor=proc,
                                      training_config=tc, **kw)
    n_tr, n_va, n_te = (len(trainer.train_indices), len(trainer.val_indices),
                        len(getattr(trainer, "test_indices", []) or []))
    print("[4/5] 划分已复现：train/val/test = %d/%d/%d" % (n_tr, n_va, n_te))
    for name, got in (("train", n_tr), ("val", n_va), ("test", n_te)):
        if got != EXPECT[name]:
            print("❌ 划分规模不一致：%s 期望 %d 实得 %d（split 文件或 CSV 版本不匹配）" % (
                name, EXPECT[name], got))
            return 2

    # 前向 + 导出（与 save_test_predictions 同口径）
    data = trainer.data if hasattr(trainer, "data") else proc.graph_data
    node_types = list(getattr(trainer, "node_types", data.node_types))
    edge_types = list(getattr(trainer, "edge_types", data.edge_types))
    x_dict = {nt: data[nt].x for nt in node_types if hasattr(data[nt], "x")}
    edge_dict = {et: data[et].edge_index for et in edge_types if hasattr(data[et], "edge_index")}
    with torch.no_grad():
        out = model(x_dict, edge_dict)
    soc_logits = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out

    rows = []
    for idx in trainer.test_indices:
        node = trainer.all_nodes[idx]
        ntype = trainer.all_node_types[idx]
        truth = trainer.all_soc_labels[idx]
        if truth < 0 or ntype not in soc_logits:
            continue
        local = trainer._get_node_index(node, ntype)
        if local is None:
            continue
        pred = int(soc_logits[ntype][local].argmax(dim=0).item())
        true_code = trainer.soc_list[truth] if truth < len(trainer.soc_list) else None
        pred_code = trainer.soc_list[pred] if pred < len(trainer.soc_list) else None
        rows.append({"node_code": int(node), "level": ntype,
                     "true_soc_code": true_code, "pred_soc_code": pred_code,
                     "correct": bool(pred == truth)})

    acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
    print("[5/5] 导出 %d 条预测，测试集 Acc = %.6f" % (len(rows), acc))
    rec_acc = ck.get("best_metrics", {}).get("test_accuracy")
    if rec_acc is None:
        rec_acc = (ck.get("test_metrics") or {}).get("accuracy")
    if rec_acc is not None and abs(acc - float(rec_acc)) > 1e-3:
        print("⚠️ 重算 Acc %.6f 与 checkpoint 记录 %.6f 相差 >1e-3，请核对 split/特征是否一致"
              % (acc, float(rec_acc)))

    out_dir = args.out_dir or ck_dir
    out = os.path.join(out_dir, "test_predictions_SAGE.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("已写入:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
