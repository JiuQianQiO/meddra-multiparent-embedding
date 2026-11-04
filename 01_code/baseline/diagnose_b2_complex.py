# -*- coding: utf-8 -*-
"""B2-ComplEx 塌缩诊断：复用 train_b2_kge 的三元组构建，逐 epoch 记录损失与嵌入范数。

回答审稿意见 M6-a："ComplEx 的 Macro-F1 仅 0.0131，是否训练塌缩？"
诊断判据：
  1. 损失曲线：若损失在早期骤降至 0 且不再下降 → 塌缩；
  2. 嵌入范数：若实体嵌入的 L2 范数随时间退化到接近 0 → 塌缩；
  3. 损失与分类指标是否背离（损失很低但分类很差）。

用法（需先 pip install pykeen）：
    python diagnose_b2_complex.py \
        --csv  D:/MedDRAData/MEDDRA_DATA_29_0.csv \
        --cache-dir D:/MedDRA模型文件/JBI/cache_shared \
        --out-dir D:/MedDRA模型文件/JBI/Baseline/B2 \
        --epochs 100 --dim 256 --seed 42
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="MedDRA 官方层级 CSV")
    ap.add_argument("--cache-dir", required=True, help="图/特征缓存目录（含 *graph.pt 与 *pubmedbert.pt）")
    ap.add_argument("--out-dir", default="./b2_diag")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["MEDDRA_CSV_PATH"] = args.csv
    os.environ["MEDDRA_CACHE_DIR"] = args.cache_dir
    os.makedirs(args.out_dir, exist_ok=True)

    import train_b2_kge as b2

    # 1) 复用主实验同款处理器 + 划分
    cfg = b2.BaselineConfig(csv_path=args.csv, cache_dir=args.cache_dir, seed=args.seed)
    print("[1/4] 构建处理器与划分")
    proc, graph = b2.build_processor(cfg)
    sp = b2.get_split(proc, graph, seed=args.seed)

    # 2) 构建三元组（与 _run_one 一致：四类父子边 + 反向，层内索引加折叠坐标偏移）
    print("[2/4] 三元组构建")
    starts, acc = {}, 0
    for nt in graph.node_types:
        starts[nt] = acc
        acc += len(graph[nt].node_codes)
    rows = []
    for (src, rel, dst) in graph.edge_types:
        ei = graph[(src, rel, dst)].edge_index
        rows += [(str(int(h) + starts[src]), f"has_{rel}", str(int(t) + starts[dst]))
                 for h, t in zip(ei[0].tolist(), ei[1].tolist())]
    entities = sorted({h for h, _, t in rows} | {t for _, _, t in rows})
    relations = sorted({r for _, r, _ in rows})
    entity_to_id = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(relations)}
    mapped = torch.tensor([[entity_to_id[h], relation_to_id[r], entity_to_id[t]]
                           for h, r, t in rows], dtype=torch.long)

    from pykeen.triples import TriplesFactory
    tf = TriplesFactory(mapped_triples=mapped, entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    print(f"    三元组 {len(mapped):,}，实体 {len(entities):,}，关系 {len(relations)}")

    # 3) PyKEEN 训练 ComplEx，逐 epoch 记录损失与嵌入范数
    print(f"[3/4] ComplEx 训练（SLCWA，dim={args.dim}, epochs={args.epochs}, seed={args.seed}）")
    b2.seed_everything(args.seed)
    import pykeen.models as _pm
    from pykeen.training import SLCWATrainingLoop
    model = _pm.ComplEx(triples_factory=tf, embedding_dim=args.dim)
    opt = torch.optim.Adam(model.get_grad_params(), lr=1e-3)
    loop = SLCWATrainingLoop(model=model, triples_factory=tf, optimizer=opt)

    # 训练前嵌入范数
    _emb0 = model.entity_representations[0](indices=None).detach().cpu().float()
    norm_before = float(_emb0.reshape(_emb0.shape[0], -1).norm().item())

    t0 = time.time()
    # 一次训练 epochs 个 epoch（逐 epoch 调用 SLCWA loop 会因内部状态不推进导致损失恒不变）
    _raw = loop.train(triples_factory=tf, num_epochs=args.epochs,
                      batch_size=args.batch_size, use_tqdm=True)
    losses = [float(x) for x in _raw]

    # 训练后嵌入范数
    _emb1 = model.entity_representations[0](indices=None).detach().cpu().float()
    norm_after = float(_emb1.reshape(_emb1.shape[0], -1).norm().item())
    norms = [norm_before, norm_after]
    print(f"    训练 {len(losses)} epochs: 首损失={losses[0]:.5f}, 末损失={losses[-1]:.5f}")
    print(f"    嵌入范数: 前={norm_before:.4f}, 后={norm_after:.4f}")

    # 4) 诊断判据
    print("[4/4] 诊断报告")
    loss_collapse = losses[-1] < 1e-3 and min(losses[-10:]) < 1e-3
    norm_decay = norms[-1] < 0.5 * norms[0]
    print(f"    首 epoch 损失={losses[0]:.5f}，末 epoch 损失={losses[-1]:.5f}")
    print(f"    首 epoch 嵌入范数={norms[0]:.4f}，末 epoch 嵌入范数={norms[-1]:.4f}")
    print(f"    损失疑似塌缩（末损失 <1e-3）: {loss_collapse}")
    print(f"    嵌入范数疑似退化（末 < 0.5×首）: {norm_decay}")

    # 冻结嵌入 + 线性探针（复用主实验协议，验证分类指标是否与损失背离）
    emb_all = model.entity_representations[0](indices=None).detach().cpu().numpy().astype(np.float32)
    if emb_all.ndim > 2:
        emb_all = emb_all.reshape(emb_all.shape[0], -1)
    fold_n = sum(len(graph[nt].node_codes) for nt in graph.node_types)
    emb = np.zeros((fold_n, emb_all.shape[1]), dtype=np.float32)
    for e, i in entity_to_id.items():
        emb[int(e)] = emb_all[i]
    proc2fold = b2.build_index_maps(proc, graph, sp)
    m, _ = b2.linear_probe(emb[proc2fold], sp["y"], sp["train"], sp["val"], sp["test"],
                           c_list=(0.25, 1.0, 4.0), seed=args.seed)
    print(f"    线性探针: {m}")

    report = {
        "model": "B2_ComplEx",
        "dim": args.dim, "epochs": args.epochs, "seed": args.seed,
        "loss_first": losses[0], "loss_last": losses[-1],
        "norm_first": norms[0], "norm_last": norms[-1],
        "loss_collapse_suspected": loss_collapse,
        "norm_decay_suspected": norm_decay,
        "probe_metrics": m,
        "losses": losses, "norms": norms,
    }
    outp = os.path.join(args.out_dir, "diagnose_B2_ComplEx.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"    诊断结果已保存: {outp}")


if __name__ == "__main__":
    main()
