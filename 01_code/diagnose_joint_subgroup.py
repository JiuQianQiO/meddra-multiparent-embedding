# -*- coding: utf-8 -*-
"""
JBI submission - offline two-dimensional subgroup diagnosis ("parent x child" cross-tabulation)

-------------------------------------------------------------------------------
What it does
-------------------------------------------------------------------------------
Table 6 of the paper needs the "joint difference", i.e. the accuracy difference between the
"parent in AND child in" cell and the "parent out AND child out" cell, which rules out the
parent-chain and child-chain label-propagation channels simultaneously. The training script
currently provides only a one-dimensional diagnosis (`_parent_supervised`), and the artefacts
store only **aggregated values** (diag_* fields of `test_metrics_*.json`) without per-node
correctness, so the quantity cannot be recovered from the artefacts afterwards.

This script fills the gap **without retraining**: the checkpoint (best_checkpoint_*.pt) already
stores
    model_state_dict / model_config / contrastive_config / training_config
    / node_types / edge_types / num_socs / level_nodes / embedding_dim / hidden_dim
    / cache_version / build_hierarchy_version / test_metrics
which is sufficient to rebuild the model offline; together with the shared graph cache and
feature cache, a **single forward pass** determines for each node whether its prediction is
correct.

-------------------------------------------------------------------------------
Four-fold self-check (each failure is flagged prominently rather than silently yielding wrong numbers)
-------------------------------------------------------------------------------
1) Reproduction check: given the same weights, the same data, and a deterministic eval/no_grad
   forward pass, the test-set Acc / Macro-F1 / parent-chain gap computed here must match the
   test_metrics recorded in the checkpoint. A mismatch sets reproduced=False and the results
   **must not be used in the paper** (it usually indicates that the feature or graph cache
   differs from the one used at training time).
2) Split check: HeteroMedDRATrainer is instantiated directly, so the split logic is identical,
   character for character, to that used at training time, and reuses its built-in
   "unified split index, per-member check" (MEDDRA_SPLIT_INDEX_FILE).
3) Configuration reconciliation: ModelConfig / ContrastiveConfig / training_config are all taken
   from the checkpoint itself; a warning is raised if num_socs disagrees with the current graph.
4) Sample-size disclosure: N is reported for every cell, and cells with too few samples are
   flagged (small cells cannot serve as evidence).

-------------------------------------------------------------------------------
Usage (Windows PowerShell)
-------------------------------------------------------------------------------
$env:MEDDRA_CSV_PATH  = 'E:/.../MEDDRA_DATA_29_0.csv'
$env:MEDDRA_CACHE_DIR = 'E:/MedDRA_model_files/cache_shared'
$env:MEDDRA_TEXT_MODEL_DIR = 'E:/hf_models/pubmedbert'   # optional

python Embedding/code/JBI/diagnose_joint_subgroup.py `
    --run-dirs 'E:/MedDRA_model_files/JBI/SAGE/C1V1' 'E:/MedDRA_model_files/JBI/SAGE/C7' `
               'E:/MedDRA_model_files/JBI/GAT' 'E:/MedDRA_model_files/JBI/NOGNN/PubMedBERT' `
    --split-index 'E:/MedDRA_model_files/JBI/split_indices.json'

Output: joint_subgroup_diag.json in each run directory; the key cells are also printed to the console.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

#  Script copies stored alongside the artefacts in the artefact directory; used by priority (character-for-character identical to that training run)
_PREFERRED = (
    "Meddra_full_trainer_SAGE.py",
    "meddra_full_trainer_GAT.py",
    "meddra_full_trainer_noGNN.py",
    "Meddra_inductive_SAGE.py",
    "Meddra_inductive_GAT.py",
    "Meddra_inductive_noGNN.py",
    "train_C7_seed43.py",
    "train_V3_seed43.py",
)


# ---------------------------------------------------------------------------
#  basic utilities
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def pick_script(run_dir: str) -> str:
    """优先取产物目录里随产物落盘的脚本副本，否则退回目录下唯一的 .py。"""
    for name in _PREFERRED:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    pys = sorted(glob.glob(os.path.join(run_dir, "*.py")))
    return pys[0] if pys else ""


def load_module(path: str, alias: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_ckpt(run_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(run_dir, "best_checkpoint*.pt")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(run_dir, "*checkpoint*.pt")))
    if not cands:
        raise FileNotFoundError(f"{run_dir} 下未找到 best_checkpoint*.pt")
    return cands[0]


def setattr_known(obj, mapping: dict) -> list:
    """只写入目标对象已存在的字段，返回实际写入的键（防止塞入陌生字段）。"""
    hit = []
    for k, v in (mapping or {}).items():
        if hasattr(obj, k):
            setattr(obj, k, v)
            hit.append(k)
    return hit


# ---------------------------------------------------------------------------
#  data and model reconstruction
# ---------------------------------------------------------------------------
class Context(object):
    """按 (脚本, 特征类型, 训练种子, 嵌入维度) 复用已建好的图与特征，避免重复构建。"""

    def __init__(self):
        self.cache = {}


def build_data(mod, args, feature_type: str, feature_model: str,
               train_seed: int, embedding_dim: int):
    """重建 HeteroData（图 + 特征）。只读缓存，不强制重建。

    【2026-10-05 修正】noGNN 三个变体各自用不同的文本编码器（PubMedBERT / SapBERT / CODER），
    特征缓存按 feature_model 分文件存放；若一律按 PubMedBERT 加载，预测将退化为 1/27 的随机水平，
    因此 feature_model 必须由断点里读取，并参与上下文缓存键，避免不同变体互相复用同一份特征。
    """
    import inspect

    dc = mod.DataConfig()
    csv = args.csv or os.environ.get("MEDDRA_CSV_PATH") or dc.meddra_csv_path
    cache_dir = args.cache_dir or os.environ.get("MEDDRA_CACHE_DIR") or dc.cache_dir
    text_dir = args.text_model_dir or os.environ.get("MEDDRA_TEXT_MODEL_DIR") or dc.text_model_dir

    if not os.path.exists(str(csv or "")):
        try:  #  auto-location identical to the main script
            found = mod._find_meddra_csv((os.getcwd(), os.path.dirname(str(mod.__file__))))
            if found:
                _log(f"    ⚠️ 配置的 CSV 不存在 → 自动定位到: {found}")
                csv = found
        except Exception:
            pass
    if not os.path.exists(str(csv or "")):
        raise FileNotFoundError("未找到 MedDRA CSV，请设 --csv 或 MEDDRA_CSV_PATH")

    dc.meddra_csv_path = csv
    dc.cache_dir = cache_dir
    dc.text_model_dir = text_dir
    #  feature_type is overridden only when the features are explicitly identified as "random features"; otherwise the script's own default is kept
    #  (SAGE/GAT default to 'pubmedbert', noGNN defaults to 'MedBERT'), so that noGNN's default convention is not broken.
    if feature_type == "random" and hasattr(dc, "feature_type"):
        dc.feature_type = "random"
    _ft = str(getattr(dc, "feature_type", feature_type) or feature_type)
    _log(f"    feature_type（实际）={_ft}{'' if not feature_model else ', feature_model=' + feature_model}")

    mod.GraphCacheManager.FEATURE_TAG = _ft
    cache_manager = mod.GraphCacheManager(dc.cache_dir)
    processor = mod.HeteroMedDRADataProcessor(dc.meddra_csv_path, cache_manager)
    graph_data, _ = processor.load_or_build_graph(
        embedding_dim=embedding_dim, force_rebuild=False, verbose=args.verbose
    )
    if graph_data is None:
        raise RuntimeError("图加载失败")

    #  [random features V3/V4] during training the random features are generated by torch.randn after _set_seed(train_seed)
    #  (via the global RNG), so the same seed must be set here before generation; otherwise these are not the random features used at training time.
    if feature_type == "random":
        mod._set_seed(int(train_seed))

    #  assemble according to the actual signature of that script (noGNN additionally accepts feature_model; SAGE/GAT do not, so it must not be passed blindly)
    fkw = {"processor": processor, "cache_manager": cache_manager,
           "embedding_dim": embedding_dim, "force_rebuild": False,
           "verbose": args.verbose, "model_local_dir": dc.text_model_dir}
    try:
        fparams = set(inspect.signature(mod.get_pubmedbert_features).parameters)
    except Exception:
        fparams = None
    if fparams is None or "feature_type" in fparams:
        fkw["feature_type"] = _ft
    if feature_model and (fparams is None or "feature_model" in fparams):
        fkw["feature_model"] = feature_model
    elif feature_model:
        _log(f"    ⚠️ 该脚本的 get_pubmedbert_features 不接受 feature_model，忽略 {feature_model}")
    features_dict, feature_dim = mod.get_pubmedbert_features(**fkw)
    if not features_dict:
        raise RuntimeError("特征生成失败")
    for ntype, feats in features_dict.items():
        if hasattr(graph_data[ntype], "x"):
            graph_data[ntype].x = feats
    return processor, graph_data, feature_dim


def build_model(mod, ck, graph_data, processor, feature_dim, device):
    """按**该脚本实际**的编码器签名重建模型。

    三个模型族签名并不一致：
        SAGE  : (node_types, edge_types, num_socs, model_config)
        GAT   : (node_types, edge_types, num_socs, model_config)
        noGNN : (node_types, num_socs, model_config)          ← 无消息传递，不需要 edge_types
    因此这里用 signature 自适应，而不是写死参数；缺任何一个无默认值的形参都会明确报错。
    """
    import inspect
    import torch

    mc_src = ck.get("model_config") or {}
    mc = mod.ModelConfig()
    wrote = setattr_known(mc, mc_src)
    missing = [k for k in ("embedding_dim", "hidden_dim") if k not in mc_src]
    if missing:
        _log(f"    ⚠️ 断点缺少 model_config 字段 {missing}，将使用脚本默认值")
    if "num_layers" not in mc_src and hasattr(mc, "num_layers"):
        _log("    提示：断点未记录 num_layers，使用脚本默认值（noGNN 等无该字段的模型属正常）")
    mc.in_channels = int(feature_dim)     #  consistent with training: use the actual feature dimension

    num_socs_ck = ck.get("num_socs")
    num_socs = len(processor.level_nodes.get("SOC", []))
    if num_socs_ck is not None and int(num_socs_ck) != int(num_socs):
        _log(f"    ⚠️ 断点 num_socs={num_socs_ck} 与当前图 {num_socs} 不一致（可能不是同一版本数据）")

    ck_nt = [str(t) for t in (ck.get("node_types") or [])]
    if ck_nt and ck_nt != [str(t) for t in graph_data.node_types]:
        _log(f"    ⚠️ 断点 node_types={ck_nt} 与当前图 {list(graph_data.node_types)} 不一致")

    #  ---- assemble constructor arguments by the actual signature (SAGE/GAT need edge_types, noGNN does not) ----
    sig = inspect.signature(mod.HeteroMedDRAEncoder.__init__)
    params = [p for p in sig.parameters if p != "self"]
    candidates = {
        "node_types": list(graph_data.node_types),
        "edge_types": list(graph_data.edge_types),
        "num_socs": int(num_socs),
        "model_config": mc,
    }
    kwargs = {k: v for k, v in candidates.items() if k in params}
    required = [p for p in params
                if sig.parameters[p].default is inspect.Parameter.empty and p not in kwargs]
    if required:
        raise RuntimeError(
            f"{mod.__name__} 的 HeteroMedDRAEncoder 需要参数 {required}，本脚本无法提供，请检查缺口")
    _log(f"    编码器构造参数: {sorted(kwargs)}（实际签名: {params}）")

    model = mod.HeteroMedDRAEncoder(**kwargs).to(device)
    try:
        model.load_state_dict(ck["model_state_dict"], strict=True)
    except Exception as exc:
        _log(f"    ❌ 权重加载失败（strict）：{exc}")
        raise
    model.eval()
    _log(f"    model_config 生效字段: {sorted(wrote)}")
    return model


# ---------------------------------------------------------------------------
#  forward pass: obtain per-node correctness
# ---------------------------------------------------------------------------
def forward_records(model, trainer, eval_indices, device):
    """返回 [(node_code, ntype, correct), ...] 以及整体预测/真值列表。口径与 evaluate() 一致。"""
    import torch

    model.eval()
    x_dict = {nt: trainer.data[nt].x for nt in trainer.node_types if hasattr(trainer.data[nt], "x")}
    edge_types = list(getattr(trainer, "edge_types", []) or [])
    edge_dict = {et: trainer.data[et].edge_index for et in edge_types
                 if hasattr(trainer.data[et], "edge_index")}
    with torch.no_grad():
        #  the forward signature of noGNN is (x_dict, edge_index_dict=None), so passing one extra argument has no side effect
        out = model(x_dict, edge_dict)
    soc_logits = out[1] if isinstance(out, (tuple, list)) else out

    by_type = {}
    for idx in eval_indices:
        by_type.setdefault(trainer.all_node_types[idx], []).append(idx)

    records, all_pred, all_truth = [], [], []
    for ntype, idxs in by_type.items():
        if ntype not in soc_logits:
            continue
        codes = [trainer.all_nodes[i] for i in idxs]
        labels = [trainer.all_soc_labels[i] for i in idxs]
        local = [trainer._get_node_index(c, ntype) for c in codes]
        pairs = [(l, lb, c) for l, lb, c in zip(local, labels, codes) if l is not None and lb >= 0]
        if not pairs:
            continue
        pos = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        truth = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
        codes_ok = [p[2] for p in pairs]
        pred = soc_logits[ntype][pos].argmax(dim=1)
        ok_list = (pred == truth).cpu().tolist()
        all_pred.extend(pred.cpu().tolist())
        all_truth.extend(truth.cpu().tolist())
        for code, ok in zip(codes_ok, ok_list):
            records.append((int(code), ntype, bool(ok)))
    return records, all_pred, all_truth


def build_children(processor, mode: str):
    """子节点映射。

    primary：由 `processor._primary_parent` 反向建「父 → 子节点」映射
             （与训练脚本的父链诊断 `_parent_supervised` 同口径：每个节点唯一主父）；
    any    ：由 `processor._parents_all` 反向建，口径更宽（任一条父子边即算子节点）。
    """
    kids = {}
    if mode == "any" and hasattr(processor, "_parents_all"):
        for code, parents in (processor._parents_all or {}).items():
            for p in (parents or []):
                kids.setdefault(int(p), set()).add(int(code))
    else:
        for code, p in (getattr(processor, "_primary_parent", {}) or {}).items():
            kids.setdefault(int(p), set()).add(int(code))
    return kids


# ---------------------------------------------------------------------------
#  two-by-two cell statistics
# ---------------------------------------------------------------------------
def _cell(cnt: dict, cor: dict, key):
    n = cnt.get(key, 0)
    c = cor.get(key, 0)
    return {"n": n, "acc": round(c / n, 6) if n else None}


def make_table(cnt, cor):
    return {
        "父∈子∈": _cell(cnt, cor, (True, True)),
        "父∈子∉": _cell(cnt, cor, (True, False)),
        "父∉子∈": _cell(cnt, cor, (False, True)),
        "父∉子∉": _cell(cnt, cor, (False, False)),
    }


def joint_diff(table):
    a, d = table["父∈子∈"]["acc"], table["父∉子∉"]["acc"]
    if a is None or d is None:
        return None
    return round((a - d) * 100, 2)


def _level_pair(cnt_in, cor_in, cnt_out, cor_out):
    return {
        "父∈train": {"n": cnt_in, "acc": round(cor_in / cnt_in, 6) if cnt_in else None},
        "父∉train": {"n": cnt_out, "acc": round(cor_out / cnt_out, 6) if cnt_out else None},
        "gap_pp": round((cor_in / cnt_in - cor_out / cnt_out) * 100, 2) if cnt_in and cnt_out else None,
    }


def analyze(records, trainer, processor, child_mode, min_cell):
    kids = build_children(processor, child_mode)
    train_set = trainer._get_train_code_set()

    #  ---- one-dimensional (for item-by-item reconciliation with the diagnosis already in the training script) ----
    p_cnt = {"in": 0, "out": 0}
    p_cor = {"in": 0, "out": 0}
    lv_stat = {}
    for code, ntype, ok in records:
        flag = trainer._parent_supervised(code, ntype)
        if flag is None:
            continue
        k = "in" if flag else "out"
        p_cnt[k] += 1
        p_cor[k] += int(ok)
        st = lv_stat.setdefault(ntype, {"cnt": {"in": 0, "out": 0}, "cor": {"in": 0, "out": 0}})
        st["cnt"][k] += 1
        st["cor"][k] += int(ok)

    #  ---- two-dimensional four cells ----
    cnt_all, cor_all = {}, {}
    cnt_hc, cor_hc = {}, {}
    lv_joint = {}
    n_nochild = 0
    for code, ntype, ok in records:
        flag = trainer._parent_supervised(code, ntype)
        if flag is None:
            continue
        my_kids = kids.get(int(code), set())
        has_child = len(my_kids) > 0
        child_in = any(int(k) in train_set for k in my_kids)
        key = (flag, child_in)
        cnt_all[key] = cnt_all.get(key, 0) + 1
        cor_all[key] = cor_all.get(key, 0) + int(ok)
        if has_child:
            cnt_hc[key] = cnt_hc.get(key, 0) + 1
            cor_hc[key] = cor_hc.get(key, 0) + int(ok)
            bucket = lv_joint.setdefault(ntype, {"cnt": {}, "cor": {}})
            bucket["cnt"][key] = bucket["cnt"].get(key, 0) + 1
            bucket["cor"][key] = bucket["cor"].get(key, 0) + int(ok)
        else:
            n_nochild += 1

    table_all = make_table(cnt_all, cor_all)
    table_hc = make_table(cnt_hc, cor_hc)

    warns = []
    for name, t in (("全诊断集", table_all), ("仅有子节点的节点", table_hc)):
        for k, v in t.items():
            if v["n"] and v["n"] < min_cell:
                warns.append(f"[{name}] 格 {k} 样本量仅 {v['n']}（< {min_cell}），不宜独立作为证据")

    return {
        "child_mode": child_mode,
        "parent_cells": _level_pair(p_cnt["in"], p_cor["in"], p_cnt["out"], p_cor["out"]),
        "parent_by_level": {
            lv: _level_pair(d["cnt"]["in"], d["cor"]["in"], d["cnt"]["out"], d["cor"]["out"])
            for lv, d in sorted(lv_stat.items())
        },
        "joint_all_diagnostic_nodes": {
            "cells": table_all,
            "joint_diff_pp": joint_diff(table_all),
            "n_nodes": int(sum(cnt_all.values())),
            "note": "完整性展示；其中「子∉」混入了无子节点的节点（LLT 全部落在该列），存在结构性偏向",
        },
        "joint_has_child_only": {
            "cells": table_hc,
            "joint_diff_pp": joint_diff(table_hc),
            "n_nodes": int(sum(cnt_hc.values())),
            "note": "推荐口径：只在有子节点的节点（HLT/PT）间比较，避免 LLT 结构性偏向「子∉」",
        },
        "joint_by_level": {
            lv: {"cells": make_table(d["cnt"], d["cor"]),
                 "joint_diff_pp": joint_diff(make_table(d["cnt"], d["cor"]))}
            for lv, d in sorted(lv_joint.items())
        },
        "n_nodes_without_child": int(n_nochild),
        "warnings": warns,
    }


# ---------------------------------------------------------------------------
#  single run directory
# ---------------------------------------------------------------------------
def process_one(run_dir, mod_cache, args, ctx, device):
    import torch

    ck_path = find_ckpt(run_dir)
    _log(f"\n{'=' * 78}\n▶ {run_dir}\n  断点: {os.path.basename(ck_path)}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    script = pick_script(run_dir)
    if not script:
        _log("    ❌ 运行目录下没有可用的训练脚本副本，跳过（可用 --script 指定）")
        return None
    key_script = os.path.abspath(script)
    if key_script not in mod_cache:
        mod_cache[key_script] = load_module(key_script, "jbi_mod_%d" % len(mod_cache))
    mod = mod_cache[key_script]
    mod.DEVICE = device            #  Trainer.__init__ moves data using a module-level DEVICE, which must be aligned

    tc = ck.get("training_config") or {}
    mc = ck.get("model_config") or {}
    embedding_dim = int(mc.get("embedding_dim", 256) or 256)
    train_seed = tc.get("train_seed") if tc.get("train_seed") is not None else tc.get("random_state", 42)
    if "test_ratio" not in tc:
        _log("    ⚠️ 断点未记录 test_ratio：划分可能与三分协议不一致，以复现校验结果为准")

    if args.feature_type == "auto":
        feature_type = "random" if int(mc.get("in_channels", 768) or 768) == 128 else "pubmedbert"
    else:
        feature_type = args.feature_type

    #  ---- text encoder (feature_model) ----
    #  checkpoint first (the field exists in noGNN's ModelConfig); then --feature-model; finally infer from the directory / checkpoint name.
    feature_model = ""
    if mc.get("feature_model"):
        feature_model = str(mc["feature_model"])
    elif args.feature_model:
        feature_model = args.feature_model
    else:
        _blob = (os.path.basename(ck_path) + " " + os.path.basename(run_dir)).upper()
        if "SAPBERT" in _blob:
            feature_model = "SapBERT"
        elif "CODER" in _blob:
            feature_model = "CODER"
        elif "PUBMEDBERT" in _blob or "PUBMED" in _blob:
            feature_model = "PubMedBERT"
    if feature_model:
        _log(f"    feature_model={feature_model}（文本编码器，决定加载哪份特征缓存）")

    _log(f"    feature_type={feature_type}（断点 in_channels={mc.get('in_channels')}）, "
         f"train_seed={train_seed}")

    cache_key = (key_script, feature_type, feature_model, int(train_seed), embedding_dim)
    if cache_key not in ctx.cache:
        _log("    构建图与特征（首次；后续同配置模型直接复用）…")
        ctx.cache[cache_key] = build_data(mod, args, feature_type, feature_model,
                                          int(train_seed), embedding_dim)
    processor, graph_data, feature_dim = ctx.cache[cache_key]
    _log(f"    特征维度={feature_dim}")

    model = build_model(mod, ck, graph_data, processor, feature_dim, device)

    #  the training / contrastive configuration is taken entirely from the checkpoint, ensuring the split uses the same parameters as during training
    training_config = mod.TrainingConfig()
    wrote_t = setattr_known(training_config, tc)
    if "test_ratio" not in wrote_t:
        _log(f"    ⚠️ TrainingConfig 未接受 test_ratio 字段（已接受: {sorted(wrote_t)}）")

    #  the noGNN script has no contrastive learning and therefore no ContrastiveConfig; adapt to whether it exists
    contrastive_config = None
    if hasattr(mod, "ContrastiveConfig"):
        contrastive_config = mod.ContrastiveConfig()
        setattr_known(contrastive_config, ck.get("contrastive_config") or {})
    else:
        _log("    提示：该脚本无 ContrastiveConfig（noGNN 无对比学习），训练器按 4 参构造")

    import inspect

    tsig = inspect.signature(mod.HeteroMedDRATrainer.__init__)
    tparams = [p for p in tsig.parameters if p != "self"]

    #  [explicitly block inductive scripts] the inductive setting uses the "entire HLT subtree hold-out" protocol: there is no three-way test set,
    #  and before holding out, select_inductive_holdout must run and edges are masked under the MASK protocol (which modifies the graph in place).
    #  its cells do not share their source with the transductive setting and cannot be mixed with Table 6, so this script does not support it and emits an actionable message.
    if "inductive_holdout" in tparams:
        raise RuntimeError(
            "该运行目录是归纳式脚本（留出整棵 HLT 子树），本诊断脚本不支持：\n"
            "  · 归纳式未做三分切分，没有 test 集，四格无法与转导式表 6 同口径；\n"
            "  · 其划分来自独立的 INDUCTIVE_SEED/留出口径，与 split_indices.json 不同源；\n"
            "  · MASK 协议需要 mask_edges_for_holdout 就地改图，会污染本脚本复用的共享图数据。\n"
            "  处理方式：本次诊断请排除 IND* 目录（论文表 6 也不需要它们）。")
    t_candidates = {
        "model": model, "data": graph_data, "processor": processor,
        "training_config": training_config, "contrastive_config": contrastive_config,
    }
    tkwargs = {k: v for k, v in t_candidates.items() if k in tparams}
    t_required = [p for p in tparams
                  if tsig.parameters[p].default is inspect.Parameter.empty and p not in tkwargs]
    if t_required:
        raise RuntimeError(f"{mod.__name__} 的 HeteroMedDRATrainer 需要参数 {t_required}，本脚本无法提供")
    _log(f"    训练器构造参数: {sorted(tkwargs)}")

    trainer = mod.HeteroMedDRATrainer(**tkwargs)

    if args.eval_set == "test":
        eval_indices = list(getattr(trainer, "test_indices", []) or [])
        if not eval_indices and getattr(trainer, "val_indices", []):
            _log("    ⚠️ 该模型没有测试集（例如归纳式脚本未转三分），改用验证集")
            eval_indices = list(trainer.val_indices)
    else:
        eval_indices = list(getattr(trainer, "val_indices", []) or [])
    if not eval_indices:
        _log(f"    ❌ {args.eval_set} 集为空，跳过")
        return None
    _log(f"    评估集={args.eval_set}  n={len(eval_indices):,}"
         f"（train {len(trainer.train_indices):,} / val {len(trainer.val_indices):,} / "
         f"test {len(getattr(trainer, 'test_indices', []) or []):,}）")

    records, preds, truths = forward_records(model, trainer, eval_indices, device)

    acc = sum(1 for r in records if r[2]) / max(1, len(records))
    try:
        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(truths, preds, average="macro", zero_division=0))
    except Exception:
        macro_f1 = None

    ref = ck.get("test_metrics") or ck.get("best_metrics") or {}
    tol = args.tolerance
    checks = {}
    if ref.get("accuracy") is not None:
        checks["accuracy"] = {"ref": round(float(ref["accuracy"]), 6), "now": round(acc, 6),
                              "ok": abs(float(ref["accuracy"]) - acc) <= tol}
    if macro_f1 is not None and ref.get("macro_f1") is not None:
        checks["macro_f1"] = {"ref": round(float(ref["macro_f1"]), 6), "now": round(macro_f1, 6),
                              "ok": abs(float(ref["macro_f1"]) - macro_f1) <= tol}

    diag = analyze(records, trainer, processor, args.child_mode, args.min_cell)
    pc = diag["parent_cells"]
    for key, refkey in (("父∈train", "diag_parent_in_train_acc"), ("父∉train", "diag_parent_out_train_acc")):
        if ref.get(refkey) is not None and pc[key]["acc"] is not None:
            checks[refkey] = {"ref": round(float(ref[refkey]), 6), "now": pc[key]["acc"],
                              "ok": abs(float(ref[refkey]) - pc[key]["acc"]) <= tol}
    for lv in ("hlt", "pt", "llt"):
        rv = ref.get(f"diag_{lv}_gap")
        nowv = diag["parent_by_level"].get(lv, {}).get("gap_pp")
        if rv is not None and nowv is not None:
            checks[f"diag_{lv}_gap"] = {"ref": round(float(rv) * 100, 2), "now": nowv,
                                        "ok": abs(float(rv) * 100 - nowv) <= tol * 100}

    reproduced = all(v.get("ok") for v in checks.values()) and bool(checks)
    _log(f"    复现校验: {'✅ 全部一致' if reproduced else '❌ 存在偏差（结果不可直接写入论文）'}")
    for k, v in checks.items():
        _log(f"       {k:<22} 断点={v['ref']}  本次={v['now']}  {'✓' if v['ok'] else '✗'}")

    out = {
        "run_dir": run_dir,
        "checkpoint": os.path.basename(ck_path),
        "script": key_script,
        "eval_set": args.eval_set,
        "eval_n": int(len(eval_indices)),
        "feature_type": feature_type,
        "feature_model": feature_model or None,
        "train_seed": int(train_seed),
        "split_seed": int(training_config._split_seed),
        "reproduce": {"checks": checks, "tolerance": tol, "reproduced": bool(reproduced)},
        "overall": {
            "accuracy": round(acc, 6),
            "macro_f1": round(macro_f1, 6) if macro_f1 is not None else None,
            "n": int(len(records)),
        },
        "joint_subgroup": diag,
    }

    dst = args.out_dir or run_dir
    os.makedirs(dst, exist_ok=True)
    #  when a unified out-dir is specified, the file name includes the run directory name so that multiple models do not overwrite each other
    _fname = ("joint_subgroup_diag.json" if not args.out_dir
              else "joint_subgroup_diag_%s.json" % os.path.basename(os.path.normpath(run_dir)))
    out_path = os.path.join(dst, _fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    _log(f"    已写出: {out_path}")

    t = diag["joint_has_child_only"]
    _log("    【二维联合差 · 仅有子节点的节点】")
    for k, v in t["cells"].items():
        _log(f"       {k}: n={v['n']:<6} acc={v['acc']}")
    _log(f"       联合差 = {t['joint_diff_pp']} 个百分点（n={t['n_nodes']}）")
    for w in diag["warnings"]:
        _log(f"    ⚠️ {w}")

    try:
        del model, trainer
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
#  entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="离线二维子群诊断（父 × 子 交叉四格），无需重训")
    ap.add_argument("--run-dirs", nargs="+", required=True, help="产物目录（内含 best_checkpoint_*.pt）")
    ap.add_argument("--eval-set", choices=("test", "val"), default="test",
                    help="在哪个集合上统计（默认 test，与表 6 口径一致）")
    ap.add_argument("--child-mode", choices=("primary", "any"), default="primary",
                    help="子节点口径：primary=以主父关系反向（与父链诊断对称，默认）；any=任一父子关系")
    ap.add_argument("--feature-type", choices=("auto", "pubmedbert", "random"), default="auto",
                    help="auto 时按断点 in_channels 判定：128→random，否则沿用脚本默认特征口径")
    ap.add_argument("--feature-model", default="",
                    help="文本编码器（noGNN 专用：PubMedBERT / SapBERT / CODER）；"
                         "缺省时优先读断点 model_config.feature_model，其次按目录/断点名推断")
    ap.add_argument("--split-index", default="",
                    help="统一划分索引文件路径（设上后每个模型都会做逐成员校验）")
    ap.add_argument("--out-dir", default="", help="结果写出目录（默认写到各自运行目录）")
    ap.add_argument("--csv", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--text-model-dir", default="")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--tolerance", type=float, default=5e-4,
                    help="复现校验容差（默认 5e-4，约合 12,041 节点中的 6 个）")
    ap.add_argument("--min-cell", type=int, default=30, help="样本量提示阈值")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.split_index:
        os.environ["MEDDRA_SPLIT_INDEX_FILE"] = args.split_index
    elif not os.environ.get("MEDDRA_SPLIT_INDEX_FILE"):
        _log("提示：未设 MEDDRA_SPLIT_INDEX_FILE。首个模型会写出划分索引，其后各模型仍与其逐成员校验；"
             "若要跨批次/跨脚本强校验，建议显式指定 --split-index。")

    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    _log(f"设备: {device}")

    mod_cache, ctx = {}, Context()
    results = []
    for d in args.run_dirs:
        d = d.strip().strip('"')
        if not os.path.isdir(d):
            _log(f"❌ 目录不存在，跳过: {d}")
            continue
        try:
            r = process_one(d, mod_cache, args, ctx, device)
        except Exception as exc:
            import traceback
            _log(f"❌ {d} 处理失败: {exc}")
            traceback.print_exc()
            continue
        if r:
            results.append(r)

    _log(f"\n{'=' * 78}")
    _log(f"完成 {len(results)}/{len(args.run_dirs)} 个运行目录")
    bad = [r["run_dir"] for r in results if not r["reproduce"]["reproduced"]]
    if bad:
        _log("以下目录复现校验未通过，**不要**直接用于论文数值：")
        for b in bad:
            _log(f"   - {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
