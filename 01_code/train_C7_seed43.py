# =============================================================================
# File: meddra_full_trainer_SAGE.py
# Purpose: MedDRA heterogeneous-graph embedding training (multi-level contrastive)
# Version: 7.2
#
# Notes:
#   1. All hyperparameters are managed centrally via @dataclass
#   2. Supports PubMedBERT feature caching to avoid recomputation;
#      additional ablation parameters are included
#   3. Full-graph training
#   4. Vectorized multi-level contrastive loss (siblings, parent-child, cousin hard negatives)
#   5. Early stopping based on validation SOC classification accuracy
# Note: the memory bank is used for incremental distillation and must preserve the old
# model's full distribution under full-graph inference, hence the full graph is used.
# Even if validation nodes participate in message passing, the representativeness of the
# memory bank is unaffected (the old model has already seen validation nodes when trained).
# Architecture note: SAGEConv explicitly preserves a node's own features through its native
# root_weight mechanism (lin_l), without relying on self-loop edges; the comparison model
# GATConv sets add_self_loops=False and compensates self-features with a residual connection.
# The two models are aligned on the "whether own features are used" dimension, which keeps
# the comparison fair.
# =============================================================================
import os
import sys
import shutil
import json
import pickle
import hashlib
import time
import gc
import warnings
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random

# =============================================================================
# 【C7 种子复核专用版】本文件由 Meddra_full_trainer_SAGE.py 整体派生，仅用于
# 复核 C7（同时移除主父兄弟与父子正样本）的超加性结果，其余实验一律拒绝：
#   - 划分种子固定 42 —— 训练/验证划分与原 C7/C1 逐成员一致（保证可比性）；
#   - 训练种子 43    —— 只换模型初始化/逐轮打乱/dropout 等训练随机性；
#   - 输出目录独立   —— <输出根>/C7-seed43，不覆盖原 C7 的任何产物；
#   - 免参数         —— 清空命令行参数，直接 `python 本文件.py` 即开始训练；
#   - CSV 自动定位   —— Kaggle /kaggle/input 优先，其次当前目录；也可设 MEDDRA_CSV_PATH。
# 可选环境变量：MEDDRA_OUTPUT_ROOT（输出根，默认 <OUTPUT_DIR> 或当前目录）、
#               MEDDRA_CACHE_DIR（共享图/特征缓存，不上传则首次自动建图）。
# 预计耗时：T4 单卡约 5.3 小时（500 轮上限内自然早停；支持断点续训）。
# =============================================================================
import sys as _sys_locked
os.environ["MEDDRA_SPLIT_SEED"] = "42"
os.environ["MEDDRA_TRAIN_SEED"] = "43"
os.environ["MEDDRA_EXPERIMENT"] = "C7"
if not os.environ.get("MEDDRA_OUTPUT_ROOT"):
    os.environ["MEDDRA_OUTPUT_ROOT"] = ("<OUTPUT_DIR>"
                                        if os.path.isdir("<OUTPUT_DIR>") else os.getcwd())
if not os.environ.get("MEDDRA_RUN_DIR"):
    os.environ["MEDDRA_RUN_DIR"] = os.path.join(os.environ["MEDDRA_OUTPUT_ROOT"], "C7-seed43")
if not os.environ.get("MEDDRA_CSV_PATH"):
    import glob as _glob_c7
    _c7_cands = []
    for _pat in ("<DATA_DIR>",
                 os.path.join(os.getcwd(), "**", "MEDDRA*.csv")):
        try:
            _c7_cands += _glob_c7.glob(_pat, recursive=True)
        except OSError:
            pass
    if _c7_cands:
        os.environ["MEDDRA_CSV_PATH"] = max(_c7_cands, key=os.path.getsize)
sys.argv = [sys.argv[0]]

# =============================================================================
#  [run directory] one folder per experiment + a shared cache directory
# -----------------------------------------------------------------------------
#  Design goal: a single run writes to exactly one folder; experiments never overwrite each other, which simplifies archiving and retrieving numbers for the paper.
#    * RUN_DIR   -- the **single output directory** of this run: logs / training-history JSON / checkpoint /
#                   weights .pth / embedding CSV / all PNG files are written into it.
#    * CACHE_DIR -- graph cache and text-feature cache, **shared across experiments** and independent of the experiment.
#                  原因：缓存只与"CSV / 维度 / 编码器"有关，与消融配置无关；若每实验各存一份，
#                        13 次运行会重复建图 13 次、重复编码 12 万个名称 13 次。
#  Directory selection priority (high -> low):
#    (1) environment variable MEDDRA_RUN_DIR                         specifies the output directory directly (the experiment name is taken from that directory name)
#    (2) environment variables MEDDRA_OUTPUT_ROOT + MEDDRA_EXPERIMENT   combine into "output root / experiment name"
#    (3) default = **current working directory**: with no path specified, logs / training history / checkpoints / weights / embedding CSV / PNG
#       and the cache are all written to the "current folder" (independent of the script name, its location, or any arguments).
#    Cache directory: MEDDRA_CACHE_DIR > <output directory>/meddra_cache
#       (to share one cache across experiments, set `MEDDRA_CACHE_DIR=<shared directory>` explicitly)
#  Usage: cd into a folder -> python <script of any name>.py              # all results land here
#       cd 到某文件夹 → python <脚本>.py C3 V4 [--jobs 2]       # 启动器：自动建 C3/ V4/
# =============================================================================
#  [Kaggle / Jupyter compatibility] __file__ is undefined inside a cell -> fall back safely to "no script path";
#  in such environments, specify the experiment name and output directory via environment variables (MEDDRA_EXPERIMENT / MEDDRA_OUTPUT_ROOT).
def _safe_script_path() -> str:
    """本模块对应的 .py 文件绝对路径；实在找不到（如 Jupyter 单元格）返回 ''。

    取值顺序（逐条校验"确实是个存在的 .py 文件"，避免拿到 `-c` / 内核启动器等假路径）：
      ① `__file__`                          —— 常规脚本运行 / 被 import 时最可靠；
      ② `sys.argv[0]`                       —— `exec(open(x).read())`、外层包装脚本注入 argv 时指向目标脚本；
      ③ `sys.modules['__main__'].__file__`  —— 前两者都取不到时的兜底（它指向的是"启动脚本"）。
    三者都不成立 → 返回 ''（调用处按"没有脚本路径"处理，例如启动器会提示改用环境变量）。
    注意：**不要用 `os.getcwd()` 顶替** —— 本函数返回的是"脚本文件"，cwd 是"启动目录"，语义不同
    （启动目录已由 `_CWD` 单独提供），且启动器要用它复制脚本，传目录会导致复制失败。
    """
    _cands = [globals().get("__file__"),
              sys.argv[0] if sys.argv and str(sys.argv[0]).endswith(".py") else None,
              getattr(sys.modules.get("__main__", None), "__file__", None)]
    for _c in _cands:
        if not _c:
            continue
        _p = os.path.abspath(str(_c))
        #  must really exist, and exclude the ipykernel launcher (it is the kernel's script, not the user's experiment script)
        if os.path.isfile(_p) and "ipykernel_launcher" not in os.path.basename(_p):
            return _p
    return ""


def _label(path) -> str:
    """目录名 → 文件名安全标签（用作实验名 / 日志名）。

    只保留字母数字与 -_. ，其余替换为 _ 。作用：无论文件夹叫什么（中文 / 空格 / 特殊字符 /
    根目录 / 被改名），生成的日志与断点文件名都合法，绝不会因为命名而报错。
    """
    _b = os.path.basename(os.path.normpath(str(path))) if path else ""
    _s = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in _b).strip("._")
    return _s or "run"


def _find_meddra_csv(roots, max_up: int = 3) -> str:
    """在给定目录及其上若干层中查找 MEDDRA*.csv（取体积最大的一个）；找不到返回 ''。

    用途：默认 CSV 路径是 Kaggle 上的绝对路径，换机器后并不存在；此函数让
    "把脚本拷到任意文件夹、什么都不配直接跑"也能自动找到数据（命中结果会写进日志）。
    """
    import glob as _glob
    _seen, _cands = set(), []
    for _r in roots:
        if not _r:
            continue
        _d = os.path.abspath(_r)
        for _ in range(max_up + 1):
            if _d in _seen or not os.path.isdir(_d):
                break
            _seen.add(_d)
            _cands += _glob.glob(os.path.join(_d, "MEDDRA*.csv"))
            _cands += _glob.glob(os.path.join(_d, "*MEDDRA*.csv"))
            _parent = os.path.dirname(_d)
            if _parent == _d:
                break
            _d = _parent
    _cands = [c for c in dict.fromkeys(_cands) if os.path.isfile(c)]
    return max(_cands, key=os.path.getsize) if _cands else ""


def _write_run_meta() -> None:
    """在输出目录写一份溯源记录：脚本路径 + 内容 md5 + 启动参数 + 关键环境变量 + 解析出的目录。

    用途：脚本被改名 / 复制多份之后，仍能确认"这份结果用的是哪版代码、什么参数"。
    任何失败都静默忽略，绝不影响训练。
    """
    try:
        _src = _safe_script_path()
        _md5 = ""
        if _src and os.path.exists(_src):
            with open(_src, "rb") as _f:
                _md5 = hashlib.md5(_f.read()).hexdigest()
        with open(os.path.join(RUN_DIR, "RUN_META.json"), "w", encoding="utf-8") as _f:
            json.dump({
                "model_tag": _MODEL_TAG,
                "experiment": EXPERIMENT,
                "run_dir": RUN_DIR,
                "cache_dir": CACHE_DIR,
                "script_source": _src,
                "script_name": os.path.basename(_src) if _src else "",
                "script_md5": _md5,
                "cwd": _CWD,
                "argv": list(sys.argv),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "env": {k: os.environ[k] for k in (
                    "MEDDRA_RUN_DIR", "MEDDRA_OUTPUT_ROOT", "MEDDRA_EXPERIMENT", "MEDDRA_CACHE_DIR",
                    "MEDDRA_CSV_PATH", "MEDDRA_TEXT_MODEL_DIR", "MEDDRA_NO_RESUME",
                    "MEDDRA_SPLIT_SEED", "MEDDRA_TRAIN_SEED") if k in os.environ},
            }, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_MODEL_TAG   = "SAGE"      #  identity of this file (used in checkpoint file names so that different models in the same directory do not overwrite each other)
_SCRIPT_PATH = _safe_script_path()
_SCRIPT_DIR  = os.path.dirname(_SCRIPT_PATH) if _SCRIPT_PATH else ""
_CWD         = os.getcwd()

if os.environ.get("MEDDRA_RUN_DIR"):
    #  (1) output directory given explicitly: the experiment name comes from that directory name (the directory name is the only source of experiment identity)
    RUN_DIR     = os.path.abspath(os.environ["MEDDRA_RUN_DIR"])
    EXPERIMENT  = os.environ.get("MEDDRA_EXPERIMENT") or _label(RUN_DIR)
    OUTPUT_ROOT = os.path.dirname(RUN_DIR)
elif os.environ.get("MEDDRA_EXPERIMENT"):
    #  (2) only the experiment name is given: output root = MEDDRA_OUTPUT_ROOT > current directory
    EXPERIMENT  = os.environ["MEDDRA_EXPERIMENT"]
    _root       = (os.path.abspath(os.environ["MEDDRA_OUTPUT_ROOT"])
                   if os.environ.get("MEDDRA_OUTPUT_ROOT") else _CWD)
    RUN_DIR     = os.path.join(_root, EXPERIMENT)
else:
    #  (3) [default] current working directory: with no path specified, logs / results / cache all land in the "current folder"
    #     (independent of the script name, the folder it sits in, or any arguments)
    RUN_DIR     = _CWD
    EXPERIMENT  = _label(RUN_DIR)

OUTPUT_ROOT = os.path.dirname(RUN_DIR)     #  for log display and backward compatibility with old configurations only
LABEL       = _label(EXPERIMENT)           #  file-name-safe tag (the experiment name may come from an environment variable and may contain illegal characters)
CACHE_DIR   = os.environ.get("MEDDRA_CACHE_DIR") or os.path.join(
    RUN_DIR, "meddra_cache")               #  the cache defaults to meddra_cache under the current directory (self-contained)
#  [launcher mode] when an experiment name is given on the command line, this process only "creates the directory + copies the script + spawns a child process";
#  it does not train itself, so it creates no run directory and writes no log file (avoiding empty directories / empty logs).
#  the launcher is enabled only when running as a .py file: in a notebook sys.argv holds kernel arguments (including paths) and cannot serve as an experiment name
_CLI_NAMES = [a for a in sys.argv[1:]
              if _SCRIPT_PATH and a and not a.startswith("-")
              and "/" not in a and "\\" not in a and "." not in a
              and not a.isdigit()]        #  exclude numbers such as in `--jobs 2`: do not mistake the job count for an experiment name
_LAUNCH_MODE = bool(_CLI_NAMES)

if not _LAUNCH_MODE:
    os.makedirs(RUN_DIR, exist_ok=True)
    _write_run_meta()      #  even a direct run leaves a provenance record (script path / md5 / arguments / resolved directories)
#  the cache directory is created **only when training actually starts** (see run_full_training): this way, a wrong argument (e.g. only `--jobs 2`
#  without an experiment name) exits with an error and leaves no empty directory on disk; the parent process in launcher mode creates no directory either.

#  [resume training] the resume file is placed in this run's output directory; its name carries the model tag
#  (when two models are occasionally run in the same directory, their checkpoints do not overwrite each other)
RESUME_FILE = os.path.join(RUN_DIR, f"resume_{_MODEL_TAG}.pt")


def _resume_paths(cfg=None):
    """解析断点文件路径 → (读取路径, 写入路径)。

    读取路径：`TrainingConfig.resume_file` 或环境变量 `MEDDRA_RESUME_FILE`（后者优先）指定时，
              使用该文件——**必须已存在**，否则告警并回落到默认；留空 → 默认（本实验目录下
              `resume_<模型标签>.pt`，即原有逻辑）。
    写入路径：**始终**是默认路径——指定文件只作为"续训种子"被读取，本流程不会覆盖或删除它。
    """
    _default = RESUME_FILE
    _want = os.environ.get("MEDDRA_RESUME_FILE") or (
        str(getattr(cfg, "resume_file", "") or "").strip() if cfg is not None else "")
    if not _want:
        return _default, _default
    _want = os.path.abspath(os.path.expanduser(_want))
    if os.path.exists(_want):
        return _want, _default
    logger.warning(f"⚠️ 配置指定的断点文件不存在：{_want} → 改用默认断点路径 {_default}")
    return _default, _default


#  set up logging: console + a file in this experiment's directory (written to both)
#  [pitfalls]
#    1) handlers=[...] is required: using only basicConfig(filename=...) makes FileHandler replace the default StreamHandler,
#       and the console stops printing;
#    2) force=True: libraries such as transformers/torch_geometric may already have attached a handler to root, in which case basicConfig
#       fails silently (no error, and no file is created);
#    3) encoding='utf-8': Windows defaults to GBK, so Chinese characters / emoji in logs raise UnicodeEncodeError;
#    4) the file name contains the experiment name + a timestamp: it distinguishes experiments and prevents a re-run from overwriting old logs;
#    5) the tqdm progress bar bypasses logging and writes straight to stderr, reaching the console only and not the file; print / exception tracebacks are likewise not captured by FileHandler.
LOG_FILE = os.path.join(RUN_DIR, f"run_{_MODEL_TAG}_{LABEL}_{time.strftime('%Y%m%d_%H%M%S')}.log")
_handlers = [logging.StreamHandler()]                       #  console (default stderr, consistent with the previous behaviour)
if not _LAUNCH_MODE:
    _handlers.append(
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')   #  log file in this experiment's directory
    )
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=_handlers,
    force=True,
)
logger = logging.getLogger(__name__)
logger.info(
    f"启动器模式：仅控制台输出，将启动 {len(_CLI_NAMES)} 个实验（{_CLI_NAMES}）" if _LAUNCH_MODE
    else f"本次日志同时写入: {LOG_FILE}"
)

warnings.filterwarnings("ignore")

# =============================================================================
#  hierarchy / label construction version -- single source of truth
# -----------------------------------------------------------------------------
#  this constant is used in three places:
#    (1) the cache key (isolating graph/feature caches of different versions);
#    (2) the version marker in cache metadata and in the checkpoint;
#    (3) "version difference detection" at load time -- a mismatch marks the cache/checkpoint invalid and triggers a rebuild.
#  [rule] whenever a change affects hierarchy_map / node_soc_map / _primary_parent / the graph structure or edges,
#         the number must be bumped; otherwise the old cache is silently reused and labels / primary parents remain on the old convention.
# =============================================================================
BUILD_HIERARCHY_VERSION = "v7.2"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resume_sanitize(obj):
    """把 numpy 标量/数组递归换成原生类型。

    用途：断点续训文件要能被 `torch.load(weights_only=True)` 读回（torch≥2.6 的默认值），
    而该模式只接受原生类型与张量——numpy 标量（如 np.float64）会被拒绝，故落盘前统一清洗。
    """
    if isinstance(obj, dict):
        return {k: _resume_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_resume_sanitize(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _safe_visual(fn, *args, **kwargs):
    """可视化失败**不得**影响已保存的训练结果（图只是附属产物）。

    动机：一次长训练曾在**收尾绘图**时因库版本差异崩溃 → 进程非零退出、启动器误报失败，
    而其实权重 / JSON / 嵌入 CSV 都已落盘。故这里只告警、不抛出。
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"⚠️ 可视化失败（不影响已保存的权重/指标/嵌入）: {type(e).__name__}: {e}")
        return None


def _make_tsne(perplexity: float = 30.0, random_state: int = 42):
    """跨 scikit-learn 版本构造 TSNE。

    ⚠️ sklearn>=1.5 把 `n_iter` 改名为 `max_iter`（用旧名直接 TypeError）。
    这里按**实际签名**选参数，避免"训练全部跑完、收尾绘图时才崩"这种代价最高的失败
    （图只是附属产物，本不该影响一次实验；另见 run_full_training 里对可视化的容错包装）。
    """
    import inspect as _inspect
    _params = _inspect.signature(TSNE.__init__).parameters
    _kw = {"n_components": 2, "random_state": random_state, "perplexity": perplexity}
    if "max_iter" in _params:
        return TSNE(**_kw, max_iter=1000)
    return TSNE(**_kw, n_iter=1000)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# =============================================================================
# 0. 配置类（Dataclass）—— 集中管理所有超参数
# =============================================================================

@dataclass
class ModelConfig:
    """
    模型架构配置类。
    
    属性:
        embedding_dim (int): 最终嵌入维度，默认 256。
            决定了每个节点最终输出的向量维度，越大表达能力越强，但计算量也越大。
            推荐范围: 128 ~ 512。
        
        hidden_dim (int): 隐藏层维度，默认 256。
            GNN 中间层的维度，通常与 embedding_dim 保持一致或略大。
            推荐范围: 128 ~ 512。
        
        dropout (float): Dropout 比率，默认 0.3。
            用于防止过拟合，在训练时随机丢弃部分神经元。
            推荐范围: 0.1 ~ 0.5。
        
        num_layers (int): GNN 层数，默认 3。
            消息传递的跳数，层数越多能捕获更远距离的邻居信息。
            对于 MedDRA 树状结构（深度为 4），2~3 层通常足够。
            推荐范围: 2 ~ 4。
        
        in_channels (int): 输入特征维度，默认 768。
            即 PubMedBERT 输出的维度，固定为 768。
            in_channels 现在由 DataConfig.feature_type 决定
    """
    embedding_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3
    num_layers: int = 3
    in_channels: int = 768  # 会被 run_full_training 覆盖


@dataclass
class TrainingConfig:
    """
    训练超参数配置类。
    
    属性:
        epochs (int): 最大训练轮数（**初始上限**），默认 500。
            注意：它同时是余弦退火的周期 T_max（CosineAnnealingLR(T_max=epochs)），
            故改它不只改轮数，还会改整条学习率轨迹 —— 跨不同 epochs 的运行不可直接比较。
            若在上限处仍未触发早停，则应延长上限后继续训练（"收敛更慢的配置获得更多轮数"）。
        
        lr (float): 初始学习率，默认 1e-3。
            控制参数更新的步长，太大会导致训练不稳定，太小则收敛过慢。
            【重要】本实现采用全批次 GD（每个 epoch 仅 1 次 optimizer.step()），
            学习率必须按“每 epoch 一次更新”的口径设置：沿用 mini-batch 的常用量级
            （1e-5 ~ 1e-4）会导致 200 轮内几乎不学习（实测 5e-5 下第 17 轮验证准确率仅
            0.2522、PT 层准确率冻结在 0.2425；而 1e-3 在第 16 轮已达 0.6641）。
            推荐范围: 5e-4 ~ 2e-3（全批次 GD 口径）。
        
        patience (int): 早停耐心值，默认 15。
            验证指标连续多少轮不提升则停止训练。
            推荐范围: 10 ~ 20。
        
        batch_size (int): 批次大小，默认 256。
            每次前向传播处理的节点数，受 GPU 显存限制。
            推荐范围: 128 ~ 1024。
        
        weight_decay (float): 权重衰减系数（L2 正则化），默认 5e-5。
            用于防止过拟合，惩罚过大的权重。
            推荐范围: 1e-5 ~ 1e-4。
        
        warmup_epochs (int): 学习率预热轮数，默认 3。
            训练初期使用较小的学习率逐步增加到目标值，有助于训练稳定。
            推荐范围: 2 ~ 5。
        
        use_contrastive (bool): 是否启用对比损失，默认 True。
            启用多层级对比学习，让兄弟节点和父子节点在嵌入空间中更接近。
            建议始终启用。
        
        val_ratio (float): 验证集比例，默认 0.1。
            按 SOC 分层采样，每个 SOC 保留该比例的节点作为验证集。
            推荐范围: 0.05 ~ 0.2。
        
        test_ratio (float): 测试集比例，默认 0.1（JBI 投稿新增的三分切分）。
            按 SOC 分层采样，先划出测试集（仅最终评估一次、不参与早停），
            再在剩余节点中按 val_ratio 划出验证集。设 0 时退化为原二分协议。
        
        random_state (int): 随机种子，默认 42。
            确保实验结果可复现。
    """
    epochs: int = 500
    lr: float = 1e-3          # 全批次 GD（每 epoch 仅 1 次参数更新）口径，见类 docstring
    patience: int = 15
    batch_size: int = 512
    weight_decay: float = 5e-5
    warmup_epochs: int = 3
    use_contrastive: bool = True
    val_ratio: float = 0.1
    test_ratio: float = 0.1   # JBI 投稿：独立测试集比例（0 = 退化为原二分协议）
    random_state: int = 42  #  hard-coded
    # 【多 seed 方差估计】把"划分种子"与"训练种子"解耦（默认 None = 沿用 random_state，行为与既有实验完全一致）：
    #   · split_seed —— 决定训练/验证划分（分层抽样的随机源）；
    #   · train_seed —— 决定模型初始化、逐轮打乱与 dropout（`train_seed*1000 + epoch` 逐轮播种）。
    #   做方差估计时**固定 split_seed=42、只改 train_seed**，从而保证验证集不变——否则换 seed 会同时
    #   换掉验证集，所得方差不能用于给既有表格做显著性判断。
    #   亦可用环境变量覆盖：MEDDRA_SPLIT_SEED / MEDDRA_TRAIN_SEED（便于批量跑而不改代码）。
    split_seed: Optional[int] = None
    train_seed: Optional[int] = None

    @property
    def _split_seed(self) -> int:
        """实际使用的划分种子（未显式指定则回落到 random_state）。"""
        return self.random_state if self.split_seed is None else int(self.split_seed)

    @property
    def _train_seed(self) -> int:
        """实际使用的训练种子（未显式指定则回落到 random_state）。"""
        return self.random_state if self.train_seed is None else int(self.train_seed)
    #  [resume training] enabled by default: if resume_<model tag>.pt exists in the run directory (and configuration / version / graph size match),
    #    training continues automatically from the next epoch; the file is deleted automatically once training ends normally (including early stopping / NaN exit).
    #    to disable: resume=False or set the environment variable MEDDRA_NO_RESUME=1.
    resume: bool = True
    resume_every: int = 10  #  how often to write the resume file (0 = never write to disk, only save at the end)
    #  [resume training · read source] empty = use resume_<model tag>.pt in this experiment's directory (previous behaviour);
    #    a path = read the checkpoint from that file (**read-only**: writing and the deletion at the end still apply to the default path,
    #    that file is neither overwritten nor deleted; it must exist, otherwise a warning is issued and the default is used).
    #    the environment variable MEDDRA_RESUME_FILE takes precedence over this field.
    resume_file: str = ""


@dataclass
class ContrastiveConfig:
    """
    对比学习超参数配置类。

    todo 可以调节 temperature 与 contrastive_weight
    
    属性:
        temperature (float): 对比损失温度系数，默认 0.2。
            控制相似度分布的锐度，温度越低，模型对正负样本的区分越严格。
            推荐范围: 0.05 ~ 0.5。
        
        contrastive_weight (float): 对比损失权重，默认 0.3。
            控制对比损失在总损失中的占比，权重过大会干扰主任务（SOC 分类）。
            推荐范围: 0.3 ~ 0.5。
        
        sibling_weight (float): 兄弟正样本权重，默认 1.0。
            兄弟节点（同一父级、同一层级）在对比损失中的权重。
            推荐范围: 0.5 ~ 2.0。

        sibling_secondary_weight (float): 【辅父】兄弟正样本权重，默认 0.3。
            仅共享辅父（非主父）的同层节点，权重降低以避免跨 SOC 假正样本。

        parent_weight (float): 父子正样本权重，默认 0.5。
            父子节点在对比损失中的权重，通常低于兄弟权重。
            推荐范围: 0.3 ~ 1.0。
        
        hard_negative_weight (float): 硬负样本权重，默认 1.5。
            表兄弟节点（同祖父级但不同父级）作为硬负样本的权重。
            高于 1.0 表示对硬负样本施加更强的惩罚。
            推荐范围: 1.0 ~ 3.0。
    """
    temperature: float = 0.2
    contrastive_weight: float = 0.3
    sibling_weight: float = 1.0
    sibling_secondary_weight: float = 0.3
    parent_weight: float = 0.5
    hard_negative_weight: float = 1.5
    # 【新增】消融开关
    use_primary_sibling: bool = True       # 主父兄弟
    use_secondary_sibling: bool = True     # 辅父兄弟
    use_parent: bool = True                #  parent-child
    use_hard_negative: bool = True         # 硬负样本
    use_mutual_exclusion: bool = True      # 正负互斥


@dataclass
class DataConfig:
    """
    数据与缓存配置类。
    
    属性:
        meddra_csv_path (str): MedDRA CSV 文件路径。
            必须包含 SOC_CODE, SOC_NAME_EN, HLGT_CODE, ... 等列。
        
        cache_dir (str): 缓存目录路径，默认 "./meddra_cache"。
            存储图数据和 PubMedBERT 特征缓存。
        
        force_rebuild_cache (bool): 是否强制重建图缓存，默认 False。
            设为 True 时忽略已有缓存，重新构建图。
        
        force_rebuild_features (bool): 是否强制重新生成 PubMedBERT 特征，默认 False。
            设为 True 时忽略已有特征缓存，重新生成。
        
        memory_k_per_soc (int): 每个 SOC 每个层级保留的记忆样本数，默认 10。
            用于构建记忆库（Memory Bank），供增量训练使用。
            推荐范围: 5 ~ 20。

        feature_type (str): 特征类型，'pubmedbert' 或 'random'
            用于消融实验，默认 'pubmedbert'
    """
    meddra_csv_path: str = "/path/to/MEDDRA_DATA_29_0.csv"
    cache_dir: str = "./meddra_cache"
    #  [new] local directory of the text encoder (PubMedBERT/SapBERT/CODER), default '' (empty).
    #    non-empty: local first, offline loading (local_files_only=True, no network access); if the directory has no weights yet,
    #         则从 HF Hub 下载一次并 save_pretrained 落盘到该目录，之后每次运行都读本地；
    #   为空：保持原行为（按 HF Hub 名称加载，依赖 HuggingFace 自身缓存）。
    #   Kaggle 用法：把已下载好的模型目录做成 Dataset，挂载后将其路径填在此处即可。
    text_model_dir: str = ""
    force_rebuild_cache: bool = False
    force_rebuild_features: bool = False
    memory_k_per_soc: int = 10
    feature_type: str = "pubmedbert"  # 新增：'pubmedbert' 或 'random'
    # [改进] 记忆库采样策略：'random' 或 'herding'
    memory_sampling_strategy: str = "herding"
    # 【新增】是否构建记忆库（Memory Bank）。其产物当前没有任何训练/推理流程消费，
    # 仅为后续增量训练预留；每次训练后构建会白花一段采样时间，故默认关闭。
    enable_memory_bank: bool = False

# =============================================================================
# 1. GPU 设备管理器
# =============================================================================

class DeviceManager:
    """
    GPU/CPU 设备管理器。
    负责检测可用设备、设置默认设备、清理显存。
    
    属性:
        device (torch.device): 当前使用的计算设备
        num_gpus (int): 可用 GPU 数量
        gpu_names (List[str]): GPU 名称列表
    """
    def __init__(self):
        self.device = self._setup_device()
        self.num_gpus = 0
        self.gpu_names = []
        self._get_gpu_info()
        logger.info(f"使用设备: {self.device}")

    def _setup_device(self) -> torch.device:
        """自动选择计算设备：优先使用 CUDA（多CUDA使用第一个，索引为0），否则使用 CPU。"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"CUDA 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA 不可用，使用 CPU")
        return device

    def _get_gpu_info(self) -> None:
        """获取并记录 GPU 详细信息（如果有）。"""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.num_gpus)]
            if self.num_gpus > 1:
                logger.info(f"检测到 {self.num_gpus} 块 GPU")
                for i, name in enumerate(self.gpu_names):
                    logger.info(f"  GPU {i}: {name}")

    def clear_memory(self) -> None:
        """清理 GPU 缓存和垃圾回收，释放显存。"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU 内存已清理")


# 全局设备实例
device_manager = DeviceManager()
DEVICE = device_manager.device


# =============================================================================
# 2. 图缓存管理器（增强版：支持 PubMedBERT 特征缓存）
# =============================================================================

class GraphCacheManager:
    """
    图数据缓存管理器，用于持久化保存已构建的异构图数据和 PubMedBERT 特征。
    避免每次运行都重新构建图和生成特征，节省时间。
    
    缓存目录结构:
        cache_dir/
            cache_info.json                      # 索引文件
            {cache_key}_graph.pt                 #  graph data
            {cache_key}_metadata.pkl             # 元数据
            {cache_key}_pubmedbert.pt            # PubMedBERT 特征
    
    属性:
        cache_dir (str): 缓存根目录路径
        cache_info (Dict[str, Dict]): 缓存索引
    """
    def __init__(self, cache_dir: str = "./meddra_cache"):
        """
        初始化缓存管理器。
        
        参数:
            cache_dir (str): 缓存目录路径，默认 "./meddra_cache"
        """
        self.cache_dir = cache_dir
        # exist_ok=True  如果目录已经存在，不报错，继续执行
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_info = {}
        # 加载缓存信息 JSON
        self._load_cache_info()
        logger.info(f"缓存管理器初始化，目录: {cache_dir}")

    def _load_cache_info(self) -> None:
        """从磁盘加载缓存索引文件 cache_info.json。"""
        info_path = os.path.join(self.cache_dir, "cache_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    self.cache_info = json.load(f)
                logger.info(f"已加载缓存索引，共 {len(self.cache_info)} 条记录")
            except Exception as e:
                logger.error(f"加载缓存索引失败: {e}")
                self.cache_info = {}

    def _save_cache_info(self) -> None:
        """将缓存索引保存到磁盘。"""
        info_path = os.path.join(self.cache_dir, "cache_info.json")
        try:
            #  [concurrency-safe] write a temporary file first and then replace atomically: writing directly would truncate files when several processes share the same cache_dir
            #  (the only consequence would be rebuilding the graph once more, but avoid it if possible; the file name carries the PID so that two processes do not clobber each other's temporary file)
            _tmp_info = f"{info_path}.tmp{os.getpid()}"
            with open(_tmp_info, "w", encoding="utf-8") as f:
                json.dump(self.cache_info, f, indent=2, ensure_ascii=False)
            try:
                os.replace(_tmp_info, info_path)
            finally:
                #  If the replacement fails (on Windows it fails when another process holds the target open; POSIX does not), leave no garbage behind
                if os.path.exists(_tmp_info):
                    try:
                        os.remove(_tmp_info)
                    except OSError:
                        pass
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")

    def _get_file_hash(self, csv_path: str) -> Optional[str]:
        """
        计算 CSV 文件的 MD5 哈希值（前12位），用于标识数据版本。
        
        参数:
            csv_path (str): CSV 文件路径
            
        返回:
            Optional[str]: 哈希字符串，若文件不存在则返回 None
        """
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return None
        try:
            with open(csv_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:12]
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            return None

    # 缓存键中的版本 = 层级/标签构建版本（单一来源，见模块顶部 BUILD_HIERARCHY_VERSION）
    CACHE_VERSION = BUILD_HIERARCHY_VERSION

    #  [feature-cache source tag] participates in the feature cache file name (default 'pubmedbert', identical to the existing cache name, so no recomputation is triggered).
    #  Assigned by run_full_training from data_config.feature_type: when a new encoder (e.g. sapbert) is added later,
    #  different feature_type values land in different cache files, avoiding "silently reusing wrong semantic features".
    FEATURE_TAG: str = "pubmedbert"
    def get_cache_key(self, csv_path: str, embedding_dim: int = 256) -> Optional[str]:
        """
        生成唯一的缓存键，基于文件哈希和嵌入维度。
        
        参数:
            csv_path (str): CSV 文件路径
            embedding_dim (int): 嵌入维度
            
        返回:
            Optional[str]: 缓存键
        """
        file_hash = self._get_file_hash(csv_path)
        if file_hash is None:
            return None
        return f"meddra_{file_hash}_{self.CACHE_VERSION}_dim{embedding_dim}"

    def has_cache(self, csv_path: str, embedding_dim: int = 256) -> bool:
        """
        检查指定 CSV 和维度是否有有效的图缓存。
        
        参数:
            csv_path (str): CSV 文件路径
            embedding_dim (int): 嵌入维度
            
        返回:
            bool: 存在完整缓存返回 True
        """
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return False
        graph_path = os.path.join(self.cache_dir, f"{key}_graph.pt")
        meta_path = os.path.join(self.cache_dir, f"{key}_metadata.pkl")
        if not (os.path.exists(graph_path) and os.path.exists(meta_path)):
            return False
        #  ---- [version difference detection] ----
        #  The version number in the file name can only "isolate": it cannot detect "changed the hierarchy/label logic but forgot to bump the number".
        #  Here the version recorded in the cache metadata is read and compared; a mismatch marks the cache invalid (returns False),
        #  and the caller takes the rebuild path, avoiding silent reuse of old labels / old primary-parent conventions.
        try:
            with open(meta_path, "rb") as f:
                cached_meta = pickle.load(f)
        except Exception as e:
            logger.warning(f"缓存元数据不可读，判定缓存失效并重建: {e}")
            return False
        cached_ver = cached_meta.get("build_hierarchy_version") if isinstance(cached_meta, dict) else None
        if cached_ver != BUILD_HIERARCHY_VERSION:
            logger.warning(
                f"⚠️ 图缓存版本不匹配：缓存={cached_ver}，当前代码={BUILD_HIERARCHY_VERSION}"
                f" → 判定缓存失效，将重建图与层级结构（旧文件保留但不使用）"
            )
            return False
        return True

    def has_pubmedbert_cache(self, csv_path: str, embedding_dim: int = 256) -> bool:
        """
        检查是否存在 PubMedBERT 特征缓存。
        
        参数:
            csv_path (str): CSV 文件路径
            embedding_dim (int): 嵌入维度
            
        返回:
            bool: 存在特征缓存返回 True
        """
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return False
        feat_path = os.path.join(self.cache_dir, f"{key}_{self.FEATURE_TAG}.pt")
        return os.path.exists(feat_path)

    def save_graph(self, csv_path: str, data: HeteroData, metadata: Dict,
                   embedding_dim: int = 256) -> bool:
        """
        保存图和元数据到缓存。
        
        参数:
            csv_path (str): 原始 CSV 路径
            data (HeteroData): 异构图数据
            metadata (Dict): 元数据字典
            embedding_dim (int): 嵌入维度
            
        返回:
            bool: 保存成功返回 True
        """
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            logger.error("无法生成缓存键，保存失败")
            return False

        graph_path = os.path.join(self.cache_dir, f"{key}_graph.pt")
        meta_path = os.path.join(self.cache_dir, f"{key}_metadata.pkl")

        try:
            torch.save(data, graph_path, _use_new_zipfile_serialization=False)
            with open(meta_path, "wb") as f:
                pickle.dump(metadata, f)
            self.cache_info[key] = {
                "csv_path": csv_path,
                "embedding_dim": embedding_dim,
                "graph_path": graph_path,
                "meta_path": meta_path,
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_nodes": getattr(data, "num_nodes", None),
                "num_edges": getattr(data, "num_edges", None),
            }
            self._save_cache_info()
            logger.info(f"图缓存已保存: {graph_path}")
            return True
        except Exception as e:
            logger.error(f"保存图缓存失败: {e}")
            return False

    """保存训练历史"""
    def save_train_history(self, csv_path: str, embedding_dim: int = 256,
                        train_history: Optional[dict] = None,
                        tag: str = "",
                        out_dir: str = "") -> bool:
        """
        保存训练历史。
        【v7.1 修复】新增 tag 参数，用于区分不同实验（SAGE-full / SAGE-noCL / τ 消融等）。
        若不加 tag，同一 CSV 的所有实验会写入同一个 JSON，互相覆盖。
        """
        if train_history is None:
            logger.error("train_history 为空，无法保存")
            return False

        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            logger.error("无法生成缓存键，训练历史保存失败")
            return False

        # ===== 转换不可序列化的数据类型 =====
        serializable_history = {}
        for k, v in train_history.items():
            if k == "hierarchical_sim_dict":
                serializable_history[k] = [
                    {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in d.items()}
                    for d in v
                ]
            else:
                if isinstance(v, list) and v and isinstance(v[0], (np.floating, float, np.integer)):
                    serializable_history[k] = [float(x) for x in v]
                else:
                    serializable_history[k] = v
        # ====================================

        #  [one folder per experiment] if out_dir is non-empty, write to this experiment's directory; if empty, fall back to the cache directory (backward compatible)
        target_dir = out_dir or self.cache_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        # 【v7.1 修复】加 tag 后缀，避免不同实验互相覆盖
        suffix = f"_{tag}" if tag else ""
        file_name = f"{key}_train_history{suffix}.json"
        file_path = os.path.join(target_dir, file_name)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            logger.info(f"训练历史已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")
            return False
        
    def load_graph(self, csv_path: str, embedding_dim: int = 256) -> Tuple[Optional[HeteroData], Optional[Dict]]:
        """
        从缓存加载图和元数据。
        
        参数:
            csv_path (str): CSV 文件路径
            embedding_dim (int): 嵌入维度
            
        返回:
            Tuple[Optional[HeteroData], Optional[Dict]]: (图数据, 元数据)
        """
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None or key not in self.cache_info:
            logger.warning(f"缓存键 {key} 不存在")
            return None, None

        info = self.cache_info[key]
        graph_path = info["graph_path"]
        meta_path = info["meta_path"]

        if not os.path.exists(graph_path) or not os.path.exists(meta_path):
            logger.warning(f"缓存文件缺失: {graph_path} 或 {meta_path}")
            return None, None

        try:
            # 添加 weights_only=False 以兼容 PyTorch 2.6+
            #  graph data
            data = torch.load(graph_path, map_location="cpu", weights_only=False)
            # metadata
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            #  [version difference detection] defensive second check (prevents calling this function directly and bypassing has_cache)
            cached_ver = metadata.get("build_hierarchy_version") if isinstance(metadata, dict) else None
            if cached_ver != BUILD_HIERARCHY_VERSION:
                logger.warning(
                    f"⚠️ 图缓存版本不匹配：缓存={cached_ver}，当前代码={BUILD_HIERARCHY_VERSION}"
                    f" → 拒绝加载该缓存，将重建"
                )
                return None, None
            logger.info(f"图缓存已加载: {graph_path}（层级/标签构建版本 {cached_ver}）")
            return data, metadata
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return None, None

    # ==================== PubMedBERT 特征缓存 ====================

    def save_pubmedbert_features(self, csv_path: str, features_dict: Dict[str, torch.Tensor],
                                 embedding_dim: int = 256) -> bool:
        """
        保存 PubMedBERT 特征到缓存。
        
        参数:
            csv_path (str): CSV 文件路径
            features_dict (Dict[str, torch.Tensor]): {node_type: (num_nodes, 768)}
            embedding_dim (int): 嵌入维度
            
        返回:
            bool: 保存成功返回 True
        """
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            logger.error("无法生成缓存键，保存失败")
            return False

        feat_path = os.path.join(self.cache_dir, f"{key}_{self.FEATURE_TAG}.pt")
        try:
            save_data = {
                'features': features_dict,
                'csv_hash': self._get_file_hash(csv_path),
                'embedding_dim': embedding_dim,
                'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'num_nodes': {k: v.shape[0] for k, v in features_dict.items()},
            }
            torch.save(save_data, feat_path)
            logger.info(f"✅ PubMedBERT 特征已缓存: {feat_path}")
            return True
        except Exception as e:
            logger.error(f"保存 PubMedBERT 特征失败: {e}")
            return False

    def load_pubmedbert_features(self, csv_path: str, embedding_dim: int = 256) -> Optional[Dict[str, torch.Tensor]]:
        """
        从缓存加载 PubMedBERT 特征。
        
        参数:
            csv_path (str): CSV 文件路径
            embedding_dim (int): 嵌入维度
            
        返回:
            Optional[Dict[str, torch.Tensor]]: 特征字典，若不存在或版本不匹配则返回 None
        """
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return None

        feat_path = os.path.join(self.cache_dir, f"{key}_{self.FEATURE_TAG}.pt")
        if not os.path.exists(feat_path):
            return None

        try:
            save_data = torch.load(feat_path, map_location="cpu", weights_only=False)
            # 验证缓存是否与当前数据匹配
            current_hash = self._get_file_hash(csv_path)
            if save_data.get('csv_hash') != current_hash:
                logger.warning("PubMedBERT 特征缓存与当前数据版本不匹配，将重新生成")
                return None
            logger.info(f"✅ PubMedBERT 特征已从缓存加载: {feat_path}")
            return save_data['features']
        except Exception as e:
            logger.error(f"加载 PubMedBERT 特征失败: {e}")
            return None

    #  [reserved / not wired] cache-list query interface: not called by any of the three current training flows (reserved for future incremental training).
    def list_cache(self) -> None:
        """打印所有缓存条目信息。"""
        if not self.cache_info:
            logger.info("暂无缓存")
            return
        logger.info("缓存列表:")
        for key, info in self.cache_info.items():
            logger.info(f"  {key} (保存时间: {info.get('save_time', '未知')})")


# =============================================================================
# 3. MedDRA 数据处理器
# =============================================================================

class HeteroMedDRADataProcessor:
    """
    MedDRA 数据处理器，负责：
        1. 加载 MedDRA CSV 文件
        2. 构建层级关系树 (SOC → HLGT → HLT → PT → LLT)
        3. 提供节点查询接口: 父级、祖父级、兄弟、SOC 等
        4. 支持版本差异检测（新增/删除节点）
        5. 创建 PyTorch Geometric 异构图 (HeteroData)
    
    核心数据结构:
        - hierarchy_map: Dict[int, Dict] = {
            node_code: {
                'parent': List[int],
                'children': List[int],
                'level': str  # 'SOC', 'HLGT', 'HLT', 'PT', 'LLT'
            }
        }
        - name_map: Dict[int, str] = {node_code: 概念名称}
        - node_soc_map: Dict[int, int] = {node_code: soc_code}  # 快速 SOC 查找
    """
    def __init__(self, meddra_csv_path: str, cache_manager: Optional[GraphCacheManager] = None):
        """
        初始化数据处理器。
        
        参数:
            meddra_csv_path (str): MedDRA CSV 文件路径
            cache_manager (Optional[GraphCacheManager]): 缓存管理器，若为 None 则自动创建
        """
        self.meddra_csv_path = meddra_csv_path
        self.cache_manager = cache_manager or GraphCacheManager()
        
        # 原始数据
        self.df: Optional[pd.DataFrame] = None
        
        # 层级结构
        self.hierarchy_map: Dict[int, Dict] = {} # {node_code: "parent": [], "children": [], "level": level}
        self.name_map: Dict[int, str] = {}      # node_code -> node_name_en (快速查找)
        self.node_soc_map: Dict[int, int] = {}  # node_code -> soc_code (快速查找)
        
        # 层级映射
        self.level_map: Dict[str, int] = {"SOC": 0, "HLGT": 1, "HLT": 2, "PT": 3, "LLT": 4}
        self.level_names: List[str] = ["SOC", "HLGT", "HLT", "PT", "LLT"]
        self.level_names_lower: List[str] = [x.lower() for x in self.level_names] # ["soc", "hlgt", "hlt", "pt", "llt"]
        
        # 各层级编码器和节点列表
        # level str: "SOC", "HLGT", "HLT", "PT", "LLT"
        self.level_encoders: Dict[str, LabelEncoder] = {} 
        self.level_nodes: Dict[str, List[int]] = {}        
        self.level_node_ids: Dict[str, List[int]] = {}
        # 每层 node 的数量
        self.num_nodes_by_level: Dict[str, int] = {}
        
        # 全局节点信息-排序后的结果
        self.num_all_nodes: int = 0
        # 有序的 全部 Node
        self.node_codes: List[int] = []
        # 有序的 全部 Node 对应的层级(字符串，不是索引)
        self.node_levels: List[str] = []
        # 有序的 全部 Node 对应的名称
        self.node_names: List[str] = []
        # 有序的 全部 Node 对应的索引 全局索引
        self.node_indices: Dict[int, int] = {}
        
        # 边类型
        self.edge_types: List[Tuple[str, str, str]] = [
            ("soc", "has", "hlgt"),
            ("hlgt", "has", "hlt"),
            ("hlt", "has", "pt"),
            ("pt", "has", "llt"),
        ]
        
        self.graph_data: Optional[HeteroData] = None
        self.build_time: Optional[str] = None
        
        #  query cache (speeds up parent-child lookups)
        self._parents_all: Dict[int, List[int]] = {}       #  full parent list (used for the primary/secondary-parent split in contrastive learning)
        self._primary_parent: Dict[int, int] = {}          #  [new] primary parent (determined from the primary row)


    # --------------------- 主流程 ---------------------

    def load_or_build_graph(self, embedding_dim: int = 256,
                            force_rebuild: bool = False,
                            verbose: bool = True) -> Tuple[Optional[HeteroData], Optional[Dict]]:
        """
        加载缓存的图，若不存在或强制重建则重新构建。
        
        参数:
            embedding_dim (int): 嵌入维度（用于缓存键）
            force_rebuild (bool): 是否忽略缓存强制重建
            verbose (bool): 是否打印详细信息
            
        返回:
            Tuple[Optional[HeteroData], Optional[Dict]]: (图数据, 元数据)
        """
        if not force_rebuild and self.cache_manager.has_cache(self.meddra_csv_path, embedding_dim):
            if verbose:
                logger.info("发现缓存的图数据，正在加载...")
            data, metadata = self.cache_manager.load_graph(self.meddra_csv_path, embedding_dim)
            if data is not None:
                self.graph_data = data
                self._restore_metadata(metadata)
                self.build_time = metadata.get("build_time", "未知")
                # _restore_metadata 内部会调用 _build_query_caches，
                # 重建 _parents_all（全部父节点）查询缓存
                return data, metadata
        if verbose:
            logger.info("未找到缓存或强制重建，开始构建图...")
        return self.build_graph(embedding_dim, verbose)

    def build_graph(self, embedding_dim: int = 256,
                    verbose: bool = True) -> Tuple[HeteroData, Dict]:
        """
        构建完整的异构图并保存到缓存。
        
        参数:
            embedding_dim (int): 嵌入维度
            verbose (bool): 是否打印进度
            
        返回:
            Tuple[HeteroData, Dict]: (图数据, 元数据)
        """
        start_time = time.time()
        # 加载 CSV 文件的数据
        self.load_data()
        # 构建数据层级，填充metadata
        self.build_hierarchy()
        # 创建异构图
        data = self.create_hetero_graph(feature_dim=128)
        
        metadata = {
            "hierarchy_map": self.hierarchy_map,
            "name_map": self.name_map,
            "node_soc_map": self.node_soc_map,
            "level_encoders": self.level_encoders,
            "level_nodes": self.level_nodes,
            "level_node_ids": self.level_node_ids,
            "num_nodes_by_level": self.num_nodes_by_level,
            "num_all_nodes": self.num_all_nodes,
            "node_codes": self.node_codes,
            "node_levels": self.node_levels,
            "node_names": self.node_names,
            "node_indices": self.node_indices,
            "edge_types": self.edge_types,
            "primary_parent": self._primary_parent,        #  [new] primary-parent mapping
            "build_hierarchy_version": BUILD_HIERARCHY_VERSION,  #  [version difference detection] single source of the version number
            "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_duration": time.time() - start_time,
        }
        
        self.cache_manager.save_graph(self.meddra_csv_path, data, metadata, embedding_dim)
        self.graph_data = data
        self.build_time = metadata["build_time"]

        # 构建查询缓存（_parents_all：全部父节点，供对比学习的主父/辅父分层使用）
        self._build_query_caches()
        
        if verbose:
            logger.info(f"图构建完成，耗时: {metadata['build_duration']:.2f} 秒")
        return data, metadata

    def build_hierarchy(self) -> Tuple[Dict, Dict]:
        """
        构建完整的层级关系树（DAG 结构）。
        
        关键修复（v7.2）：
            1. PT 的 SOC 使用官方字段 PT_SOC_CODE（不是行级 SOC_CODE）
            2. LLT 的 SOC 继承父 PT 的 PT_SOC_CODE
            3. HLT/HLGT 通过子 PT 投票聚合得到 SOC（用于分层采样与相似度分析）
            4. PRIMARY_SOC_FG 仅用于确定主父，不用于打 SOC 标签
        """
        logger.info("构建 MedDRA 层级结构（DAG）...")

        #  1. Create a LabelEncoder for each level
        for level in self.level_names:
            col = f"{level}_CODE"
            if col in self.df.columns:
                codes = self.df[col].dropna().astype(int).unique().tolist()
                codes = sorted(codes)  #  Align with the order of the LabelEncoder
                enc = LabelEncoder()
                enc.fit(codes)
                self.level_encoders[level] = enc
                self.level_nodes[level] = codes
                self.num_nodes_by_level[level] = len(codes)
                logger.info(f"  {level}: {len(codes)} 个节点")

        #  2. Iterate over all rows and establish parent-child relations (without distinguishing Y/N, preserving DAG completeness)
        self.hierarchy_map = {}
        self.name_map = {}
        self.node_soc_map = {}
        self._primary_parent = {}

        primary_count = 0
        non_primary_count = 0

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="构建层级"):
            try:
                llt = self._safe_int(row, "LLT_CODE")
                pt = self._safe_int(row, "PT_CODE")
                hlt = self._safe_int(row, "HLT_CODE")
                hlgt = self._safe_int(row, "HLGT_CODE")
                soc = self._safe_int(row, "SOC_CODE")
                pt_soc = self._safe_int(row, "PT_SOC_CODE")  #  * official primary SOC of the PT

                #  [key] record the official SOC of the PT from PT_SOC_CODE (setdefault keeps the first value; it should be unique per PT)
                if pt is not None and pt_soc is not None:
                    self.node_soc_map.setdefault(pt, pt_soc)

                #  Primary-parent record (used only for contrastive learning, not for SOC labels)
                primary_flag = str(row.get("PRIMARY_SOC_FG", "N")).strip().upper()
                if primary_flag == "Y":
                    if llt is not None and pt is not None:
                        self._primary_parent.setdefault(llt, pt)
                    if pt is not None and hlt is not None:
                        self._primary_parent.setdefault(pt, hlt)
                    if hlt is not None and hlgt is not None:
                        self._primary_parent.setdefault(hlt, hlgt)
                    if hlgt is not None and soc is not None:
                        self._primary_parent.setdefault(hlgt, soc)
                    primary_count += 1
                else:
                    non_primary_count += 1

                #  Parent-child relations (all rows, preserving the DAG)
                self._add_relationship(llt, pt, "LLT")
                self._add_relationship(pt, hlt, "PT")
                self._add_relationship(hlt, hlgt, "HLT")
                self._add_relationship(hlgt, soc, "HLGT")

                #  name
                self._store_names(row, llt, "LLT")
                self._store_names(row, pt, "PT")
                self._store_names(row, hlt, "HLT")
                self._store_names(row, hlgt, "HLGT")
                self._store_names(row, soc, "SOC")
            except Exception as e:
                logger.warning(f"跳过一行：{e}")
                continue

        logger.info(f"主 SOC 行数: {primary_count}, 非主 SOC 行数: {non_primary_count}")

        #  3. Assemble the global node list
        nodes = sorted(list(self.hierarchy_map.keys()))
        self.node_codes = nodes
        self.node_levels = [self.hierarchy_map[n]["level"] for n in nodes]
        self.node_names = [self.name_map.get(n, str(n)) for n in nodes]
        self.node_indices = {n: i for i, n in enumerate(nodes)}
        self.num_all_nodes = len(nodes)

        # 3.5 按层级整理节点列表 - 存储节点全局索引
        for level in self.level_names:
            self.level_node_ids[level] = [
                self.node_indices[n] for n in nodes if self.hierarchy_map[n]["level"] == level
            ]
        logger.info(f"层级结构构建完成: 总节点数 {self.num_all_nodes}")

        #  4. PT already obtained its SOC label from PT_SOC_CODE
        pt_with_label = sum(1 for pt in self.level_nodes.get("PT", []) if pt in self.node_soc_map)
        logger.info(f"PT 官方 SOC 标签: {pt_with_label}/{len(self.level_nodes.get('PT', []))}")

        #  5. LLT inherits the SOC of its parent PT
        llt_inherited = 0
        for llt_code in self.level_nodes.get("LLT", []):
            parents = self.hierarchy_map.get(llt_code, {}).get("parent", [])
            # 获取第一个，也是唯一一个父亲
            if parents and parents[0] in self.node_soc_map:
                self.node_soc_map[llt_code] = self.node_soc_map[parents[0]]
                llt_inherited += 1
        logger.info(f"LLT SOC 标签（继承父 PT）: {llt_inherited}/{len(self.level_nodes.get('LLT', []))}")

        #  6. HLT is aggregated by voting over child PTs
        for hlt_code in self.level_nodes.get("HLT", []):
            children_pt = self.hierarchy_map.get(hlt_code, {}).get("children", [])
            votes = [self.node_soc_map[c] for c in children_pt if c in self.node_soc_map]
            if votes:
                # Counter(votes)：统计每个 SOC 值出现的次数 Counter({"SOC_A": 2, "SOC_B": 1})
                # .most_common(1)：取出出现次数最多的 1 个，返回列表 [("SOC_A", 2)]
                # [0][0]：取第一个元组的第一个元素，也就是 "SOC_A"
                self.node_soc_map[hlt_code] = Counter(votes).most_common(1)[0][0]

        #  7. HLGT is aggregated by voting over child HLTs
        for hlgt_code in self.level_nodes.get("HLGT", []):
            children_hlt = self.hierarchy_map.get(hlgt_code, {}).get("children", [])
            votes = [self.node_soc_map[c] for c in children_hlt if c in self.node_soc_map]
            if votes:
                self.node_soc_map[hlgt_code] = Counter(votes).most_common(1)[0][0]

        #  8. SOC label coverage
        # 所有不是 SOC 的节点
        non_soc_nodes = [n for n in self.node_codes
                        if self.hierarchy_map[n]["level"] != "SOC"]
        # 所有不是 SOC 的节点，并且有 SOC 标签
        labeled_nodes = [n for n in non_soc_nodes if n in self.node_soc_map]
        logger.info(f"SOC 标签覆盖率: {len(labeled_nodes)}/{len(non_soc_nodes)} "
                    f"({len(labeled_nodes)/len(non_soc_nodes)*100:.2f}%)")

        #  9. SOC distribution diagnostics (key)
        # Counter 是一个“计数器”，用来统计一堆元素里每个元素出现了多少次； 是一个字典，键是元素，值是出现次数
        soc_counts = Counter(self.node_soc_map.values())
        total = sum(soc_counts.values())
        logger.info("=" * 60)
        logger.info("SOC 标签分布（Top 5）:")
        # most_common(n)：返回出现次数最多的前 n 个，格式是 [(元素, 次数), ...]
        for soc, cnt in soc_counts.most_common(5):
            logger.info(f"  SOC {soc} ({self.get_node_name(soc)}): {cnt} ({cnt/total*100:.1f}%)")
        logger.info(f"最大 SOC 占比: {soc_counts.most_common(1)[0][1] / total * 100:.1f}%")
        logger.info("=" * 60)

        #  10. Primary-parent coverage
        # 所有不是 SOC 的节点，并且有主父标签
        has_primary = [n for n in non_soc_nodes if n in self._primary_parent]
        logger.info(f"主父覆盖率: {len(has_primary)}/{len(non_soc_nodes)} "
                    f"({len(has_primary)/len(non_soc_nodes)*100:.2f}%)")

        return self.hierarchy_map, self.name_map

    def _restore_metadata(self, metadata: Dict) -> None:
        for key in [
            "hierarchy_map", "name_map", "node_soc_map", "level_encoders",
            "level_nodes", "level_node_ids", "num_nodes_by_level",
            "num_all_nodes", "node_codes", "node_levels",
            "node_names", "node_indices",
        ]:
            setattr(self, key, metadata.get(key, {} if key != "num_all_nodes" else 0))

        # 恢复主父映射
        self._primary_parent = metadata.get("primary_parent", {})

        self._build_query_caches()

    # --------------------- 数据加载和层级构建 ---------------------

    def load_data(self) -> pd.DataFrame:
        logger.info(f"加载 MedDRA 数据从: {self.meddra_csv_path}")
        if not os.path.exists(self.meddra_csv_path):
            raise FileNotFoundError(f"文件不存在: {self.meddra_csv_path}")
        self.df = pd.read_csv(self.meddra_csv_path)
        logger.info(f"数据形状: {self.df.shape}")
        
        #  新增：验证 PRIMARY_SOC_FG 字段存在且格式正确
        if "PRIMARY_SOC_FG" not in self.df.columns:
            logger.warning("⚠️ PRIMARY_SOC_FG 字段不存在，将使用第一个 SOC 作为标签")
        else:
            unique_flags = self.df["PRIMARY_SOC_FG"].dropna().unique()
            logger.info(f"PRIMARY_SOC_FG 取值: {unique_flags}")
            n_primary = (self.df["PRIMARY_SOC_FG"] == "Y").sum()
            logger.info(f"主 SOC 行数: {n_primary} / {len(self.df)}")
        
        # 诊断多父结构
        self.diagnose_multi_parent()
        
        return self.df

    # --------------------- 辅助方法 ---------------------

    #  [deprecated / DEPRECATED] label construction now uses the v7.2 hierarchical strategy (PT uses the official PT_SOC_CODE,
    #  LLT inherits from its parent PT, HLT/HLGT are aggregated by voting over children); this method is no longer called
    #  in any of the three training scripts after v7.2 and is kept only for backward compatibility (see the archived script meddra/old/ for its call sites).
    def _infer_missing_soc_labels(self) -> None:
        """
        [已废弃·无调用] 为未获得 SOC 标签的节点（HLT/HLGT）从父节点继承 SOC。
        
        背景：MedDRA 29.0 中有 184 个节点（173 HLT + 11 HLGT）从未以
            PRIMARY_SOC_FG='Y' 出现，但它们有有效的 SOC 关联。
            本方法按层级从高到低（HLGT → HLT）回退到父节点 SOC。
        
        对多父节点：取第一个有效父节点的 SOC（与对比学习的“主父优先”取法一致）。
        """
        level_order = {"HLGT": 1, "HLT": 2, "PT": 3, "LLT": 4}
        
        unlabeled = [
            n for n in self.node_codes
            if self.hierarchy_map[n]["level"] != "SOC"
            and n not in self.node_soc_map
        ]
        if not unlabeled:
            return
        
        #  Sort levels from high to low so that parent SOCs are inferred first (HLGT before HLT)
        unlabeled.sort(key=lambda n: level_order.get(self.hierarchy_map[n]["level"], 99))
        
        inferred = 0
        failed = []
        for node in unlabeled:
            parents = self.hierarchy_map.get(node, {}).get("parent", [])
            found = False
            for p in parents:
                if p in self.node_soc_map:
                    self.node_soc_map[node] = self.node_soc_map[p]
                    inferred += 1
                    found = True
                    break
                #  The direct parent of an HLGT is the SOC itself
                if self.hierarchy_map.get(p, {}).get("level") == "SOC":
                    self.node_soc_map[node] = p
                    inferred += 1
                    found = True
                    break
            if not found:
                failed.append(node)
        
        logger.info(f"✅ SOC 标签推断完成: {inferred}/{len(unlabeled)} 个节点从父节点继承 SOC")
        if failed:
            failed_by_level = Counter(self.hierarchy_map[n]["level"] for n in failed)
            logger.warning(f"⚠️ 仍有 {len(failed)} 个节点无法推断 SOC: {dict(failed_by_level)}")



    def _safe_int(self, row: pd.Series, col: str) -> Optional[int]:
        """安全地将列值转为 int，若为 NaN 则返回 None。"""
        val = row.get(col)
        return int(val) if pd.notna(val) else None


    #  [deprecated / DEPRECATED] since v7.2 the SOC mapping is written directly into self.node_soc_map by build_hierarchy;
    #  this method is no longer called in any of the three training scripts (see the archived script meddra/old/).
    def _set_node_soc_code(self, node_code, soc_code, overwrite=False):
        """
        [已废弃·无调用] 安全地存储节点到 SOC 的映射。
        
        参数:
            node_code: 节点代码
            soc_code: SOC 代码
            overwrite (bool): 是否覆盖已有映射。默认 False（保留首次赋值）
                            仅在处理 PRIMARY_SOC_FG='Y' 行时调用，因此默认不覆盖
                            可保证多父节点只保留主 SOC 标签
        """
        if node_code is None or soc_code is None:
            return
        if not overwrite and node_code in self.node_soc_map:
            return  # 保留已有的主 SOC 标签
        self.node_soc_map[node_code] = soc_code


    def _add_relationship(self, child: Optional[int], parent: Optional[int],
                          child_level: str) -> None:
        """添加父子关系到 hierarchy_map。"""
        if child is None or parent is None:
            return
        for node in [child, parent]:
            if node not in self.hierarchy_map:
                if node == child:
                    level = child_level
                else:
                    level = self._get_parent_level(child_level)
                # 创建空对象 
                self.hierarchy_map[node] = {"parent": [], "children": [], "level": level}

        # 设置值 
        if parent not in self.hierarchy_map[child]["parent"]:
            self.hierarchy_map[child]["parent"].append(parent)

        if child not in self.hierarchy_map[parent]["children"]:
            self.hierarchy_map[parent]["children"].append(child)

    def _get_parent_level(self, child_level: str) -> str:
        """根据子层级获取父层级名称。"""
        order = ["LLT", "PT", "HLT", "HLGT", "SOC"]
        idx = order.index(child_level)
        return order[idx + 1] if idx + 1 < len(order) else "SOC"

    def _store_names(self, row: pd.Series, code: Optional[int], level: str) -> None:
        """存储概念名称到 name_map。首先获取英文名称"""
        if code is None:
            return
        name = row.get(f"{level}_NAME_EN")
        if pd.isna(name):
            name = row.get(f"{level}_NAME")
        if pd.notna(name):
            self.name_map[code] = str(name)

    # --------------------- 构建异构图 ---------------------

    def create_hetero_graph(self, feature_dim: int = 128) -> HeteroData:
        """创建 PyTorch Geometric 的 HeteroData 对象。"""
        logger.info("创建异构图...")
        data = HeteroData()

        # 1. 设置每层节点特征
        # 遍历大写的层级名称
        for level in self.level_names:
            # 小写的层级名称
            lvl = level.lower()
            #  all nodes at the current level
            nodes = self.level_nodes.get(level, [])
            if not nodes:
                continue
            # 设置随机特征  纯高斯噪声  创建临时随机特征 保证结构的完整性，后序会被 PubMedBERT或者Random替换
            # 创建一个矩阵 行数 = len(nodes)（节点数量） 列数 = feature_dim（特征维度，默认128）
            data[lvl].x = torch.randn(len(nodes), feature_dim)
            # 取值： data['hlgt'].x[0]  # hlgt层第1个节点的128维特征向量
            data[lvl].node_codes = nodes
            logger.info(f"    {level}: {len(nodes)} 个节点")


        # 2. 设置边 父-子
        edge_pairs = [("SOC", "HLGT"), ("HLGT", "HLT"), ("HLT", "PT"), ("PT", "LLT")]
        for parent_level, child_level in edge_pairs:
            parent_lower, child_lower = parent_level.lower(), child_level.lower()

            # 获取层级节点编码器
            # HeteroData 中每个节点类型是独立存储的
            # 边索引必须使用各层内部的局部索引
            parent_enc = self.level_encoders[parent_level]
            child_enc = self.level_encoders[child_level]
            
            edges = []
            # 层级所有节点
            parent_nodes = self.level_nodes.get(parent_level, [])
            child_set = set(self.level_nodes.get(child_level, []))

            # 遍历所有父节点
            for parent in parent_nodes:
                if parent not in self.hierarchy_map:
                    continue
                # 遍历父节点的所有子节点
                p_idx = parent_enc.transform([parent])[0]
                for child in self.hierarchy_map[parent].get("children", []):
                    # 校验节点正确
                    if child in child_set:
                        # 使用对应的层级编码器获取层级索引
                        c_idx = child_enc.transform([child])[0]
                        edges.append([p_idx, c_idx])
            
            if edges:
                # 将边数组转化为张量,并且进行转置和内存连续
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                # 为图设置边 edge_index属性存储边数据
                data[parent_lower, "has", child_lower].edge_index = edge_tensor
                logger.info(f"    {parent_level}→{child_level}: {len(edges)} 条边")
        
        # 3. 添加反向边（使图无向，便于消息传递）
        for edge_type in list(data.edge_types):
            # 获取边类型
            src, rel, dst = edge_type
            # 反转边类型及数据
            reverse_type = (dst, rel, src)
            if reverse_type not in data.edge_types and hasattr(data[edge_type], "edge_index"):
                data[reverse_type].edge_index = torch.stack(
                    # 对两个行向量(data[edge_type].edge_index[1/0])进行 行(dim=0)堆叠
                    [data[edge_type].edge_index[1], data[edge_type].edge_index[0]], dim=0
                )


        # 新增：验证 DAG 结构边数统计
        logger.info("DAG 结构边统计：")
        for parent_level, child_level in edge_pairs:
            parent_lower = parent_level.lower()
            child_lower = child_level.lower()
            edge_key = (parent_lower, "has", child_lower)
            if edge_key in data.edge_types and hasattr(data[edge_key], "edge_index"):
                num_edges = data[edge_key].edge_index.shape[1]
                logger.info(f"  {parent_level}→{child_level}: {num_edges} 条边")
        
        logger.info(f"异构图构建完成! 节点类型: {list(data.node_types)}")
        return data

    def diagnose_multi_parent(self) -> Dict[str, Dict]:
        """
        诊断多父节点统计，用于论文数据报告与代码验证。
        
        返回:
            Dict[str, Dict]: 各层级的多父统计
                {
                    "HLGT": {"total": 337, "multi_parent": 17, "ratio": 0.0504, "dist": {1: 320, 2: 17}},
                    "HLT": {...},
                    "PT": {...},
                    "LLT": {...},
                }
        """  

             
        results = {}
        for level, col in [("HLGT", "HLGT_CODE"), ("HLT", "HLT_CODE"),
                        ("PT", "PT_CODE"), ("LLT", "LLT_CODE")]:
            # 收集每个节点的父节点
            # 参数：set 集合  无序不重复元素集合
            node_parents = defaultdict(set)
            
            # 根据层级确定其父层级列
            parent_col = {
                "HLGT": "SOC_CODE",
                "HLT": "HLGT_CODE",
                "PT": "HLT_CODE",
                "LLT": "PT_CODE",
            }[level]
            
            for _, row in self.df.iterrows():
                child = self._safe_int(row, col)
                parent = self._safe_int(row, parent_col)
                if child is not None and parent is not None:
                    node_parents[child].add(parent)
            
            # 统计
            total = len(node_parents)
            multi_parent = sum(1 for p in node_parents.values() if len(p) > 1)
            # 统计父节点数量分布
            dist = defaultdict(int)
            for parents in node_parents.values():
                dist[len(parents)] += 1
            
            results[level] = {
                "total": total,
                "multi_parent": multi_parent,
                "ratio": multi_parent / total if total > 0 else 0.0,
                "dist": dict(sorted(dist.items())),
            }
        
        # 打印统计
        logger.info("=" * 60)
        logger.info("MedDRA 多父节点统计 (DAG 结构)")
        logger.info("=" * 60)
        for level, stats in results.items():
            logger.info(f"  {level}: {stats['multi_parent']}/{stats['total']} "
                        f"({stats['ratio']*100:.2f}%) 多父节点")
            logger.info(f"       父节点数分布: {stats['dist']}")
        
        return results

    # --------------------- 层级查询接口 ---------------------

    def _build_query_caches(self):
        """
        构建父节点缓存（用于对比学习中的父子/表兄弟关系）。
        
        多父节点处理说明：
            MedDRA 是 DAG 结构，存在显著多父现象：
            - HLGT: 17/337 (5.04%) 有 2 个 SOC 父节点
            - HLT:  18/1739 (1.04%) 有 2 个 HLGT 父节点
            - PT:   10350/27361 (37.83%) 有 2~7 个 HLT 父节点
            - LLT:  0/91082 (0%) 唯一父节点
            
            对比学习的父子正样本和表兄弟硬负样本需要唯一父节点。
            v7.1 起，“主父”优先取 _primary_parent（PRIMARY_SOC_FG='Y' 确定的主父），
            未记录主父时才回退到 parents[0]；该取法由调用处直接读取 _primary_parent 完成，
            不再单独缓存（原 _parent_cache 已随 get_parent 一并移除）。
            
            对 PT 层的影响：37.83% 的 PT 仅与其中一个父 HLT 建立对比关系，
            但这不影响 SOC 分类任务（标签已通过 PRIMARY_SOC_FG 唯一确定）。
            
            层级相似度分析中（compute_hierarchical_intra_similarity），
            仍使用 hierarchy_map 中的完整父节点列表，不受此缓存影响。

            v7.1 新增：
            - _parents_all: 所有父节点列表（用于主父/辅父分层）

        """
        self._parents_all = {}

        for node in self.node_codes:
            self._parents_all[node] = self.hierarchy_map.get(node, {}).get("parent", [])


    def get_soc_node_code(self, node_code: int) -> Optional[int]:
        """
        快速获取节点所属的 SOC 代码 。
        
        参数:
            node_code (int): 节点代码
            
        返回:
            Optional[int]: SOC 代码，若不存在则返回 None
        """
        if node_code is None:
            return None
        return self.node_soc_map.get(node_code, None)

    def get_node_level(self, node_code: int) -> str:
        """获取节点所属层级名称。"""
        return self.hierarchy_map.get(node_code, {}).get("level", "UNKNOWN")

    def get_node_name(self, node_code: int) -> str:
        """获取节点概念名称。"""
        return self.name_map.get(node_code, str(node_code))

    def get_nodes_by_level(self, level: str) -> List[int]:
        """获取指定层级的所有节点代码。"""
        return self.level_nodes.get(level, [])

    def get_node_names_dict(self) -> Dict[str, List[str]]:
        """获取所有层级的节点名称字典。
           PubMedBERT 中对名称进行编码
        """
        return {
            l.lower(): [self.get_node_name(n) for n in self.level_nodes.get(l, [])]
            for l in self.level_names
        }

    # --------------------- 版本差异检测 ---------------------

    def get_all_node_codes(self) -> Set[int]:
        """返回所有节点代码的集合。"""
        return set(self.node_codes)

    #  [reserved / not wired] new-vs-old node difference detection: not called by the current training flow (reserved for future incremental training).
    def get_new_nodes(self, old_processor: "HeteroMedDRADataProcessor") -> List[int]:
        """
        与旧版本处理器比较，返回新增的节点代码。
        
        参数:
            old_processor (HeteroMedDRADataProcessor): 旧版本处理器
            
        返回:
            List[int]: 新节点代码列表
        """
        old_codes = old_processor.get_all_node_codes()
        return [n for n in self.node_codes if n not in old_codes]

    #  [reserved / not wired] new-vs-old node difference detection: not called by the current training flow (reserved for future incremental training).
    def get_removed_nodes(self, old_processor: "HeteroMedDRADataProcessor") -> List[int]:
        """
        与旧版本处理器比较，返回被移除的节点代码。
        
        参数:
            old_processor (HeteroMedDRADataProcessor): 旧版本处理器
            
        返回:
            List[int]: 被移除的节点代码列表
        """
        new_codes = self.get_all_node_codes()
        return [n for n in old_processor.node_codes if n not in new_codes]


# =============================================================================
# 4. 异构编码器（模型定义）
# =============================================================================

# 任何继承 nn.Module 的类，都可以通过实例调用的方式执行 forward() 方法
# model = HeteroMedDRAEncoder(...)
# output = model(x_dict, edge_dict)  #  调用 __call__ → forward()
# 不一定必须叫 model，名字不固定
# 方式二：直接调用（技术上可行但不推荐）会完全绕过 __call__ 方法，从而丢失 PyTorch 在 __call__ 中注入的所有核心功能。
#     output = model.forward(x_dict, edge_dict)
# 设计说明：
#         SAGEConv 通过其固有 root_weight 机制（即实现中的 lin_l 项 W_L · x_i）
#         显式保留节点自身特征，无需依赖 add_self_loops 参数。
#         与 GATConv 对比时，GATConv 设置 add_self_loops=False 并通过残差连接
#         补偿自身特征，从而保证两种模型在"是否使用自身特征"这一维度上等价。
class HeteroMedDRAEncoder(nn.Module):
    """
    异构 MedDRA 编码器。
    使用 SAGEConv 在异构图 (SOC→HLGT→HLT→PT→LLT) 上进行消息传递，
    生成每个节点的嵌入表示。

    架构:
        1. 节点特征投影层：将 768 维 PubMedBERT 特征投影到 hidden_dim
        2. 多层异构 SAGEConv：在异构图上传导消息
        3. 输出归一化：LayerNorm
        4. SOC 分类头：用于主任务（SOC 分类）
        5. 投影头：用于对比学习（将嵌入投影到另一空间）
    """
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        num_socs: int,
        model_config: ModelConfig,
    ):
        """
        初始化编码器。

        参数:
            node_types (List[str]): 节点类型列表，如 ['soc', 'hlgt', 'hlt', 'pt', 'llt']
            edge_types (List[Tuple[str, str, str]]): 边类型列表
            num_socs (int): SOC 类别总数
            model_config (ModelConfig): 模型架构配置
        """
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.embedding_dim = model_config.embedding_dim
        # 隐藏层维度
        self.hidden_dim = model_config.hidden_dim
        # SAGEConv 层数量
        self.num_layers = model_config.num_layers
        self.num_socs = num_socs

        # ---- 1. 节点特征投影层 ---- 节点
        # 每种节点都设置投影层
        # 每层的名称格式
        # node_lin.soc.0.weight          # Linear: 768→256
        # node_lin.soc.0.bias
        # node_lin.soc.1.weight          # LayerNorm (256)
        # node_lin.soc.1.bias
        self.node_lin = nn.ModuleDict()
        for nt in node_types:
            self.node_lin[nt] = nn.Sequential(
                Linear(model_config.in_channels, model_config.hidden_dim),
                nn.LayerNorm(model_config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(model_config.dropout),
            )

        # ---- 2. 异构 SAGEConv 层 ---- 通过边来对节点特征进行增强
        # 每层中，每种边都设置 异构 SAGEConv 层
        # 层名称格式
        # convs.0.convs.(SOC, has, HLGT).lin_l  # SAGEConv 的线性层
        # convs.0.convs.(SOC, has, HLGT).lin_r
        # convs.0.convs.(HLGT, has, HLT).lin_l
        # convs.0.convs.(HLGT, has, HLT).lin_r
        # ... (4 种边类型 × 每层)
        self.convs = nn.ModuleList()

        # 第一层：hidden_dim -> hidden_dim
        conv1 = {et: SAGEConv(model_config.hidden_dim, model_config.hidden_dim, aggr="mean") for et in edge_types}
        self.convs.append(HeteroConv(conv1, aggr="mean"))

        # 中间层（如果 num_layers > 2） hidden_dim -> hidden_dim
        for _ in range(max(0, model_config.num_layers - 2)):
            conv_mid = {et: SAGEConv(model_config.hidden_dim, model_config.hidden_dim, aggr="mean") for et in edge_types}
            self.convs.append(HeteroConv(conv_mid, aggr="mean"))

        # 最后一层：hidden_dim -> embedding_dim
        conv_last = {et: SAGEConv(model_config.hidden_dim, model_config.embedding_dim, aggr="mean") for et in edge_types}
        self.convs.append(HeteroConv(conv_last, aggr="mean"))

        #  ---- 3. output normalization ----
        # output_norm.weight   # (256,)
        # output_norm.bias     # (256,)
        self.output_norm = nn.LayerNorm(model_config.embedding_dim)

        # ---- 4. SOC 分类头 ----
        # soc_classifier.weight   # (num_socs, 256)  # num_socs ≈ 27 个 SOC
        # soc_classifier.bias     # (num_socs,)
        self.soc_classifier = nn.Linear(model_config.embedding_dim, num_socs)

        # ---- 5. 投影头（对比学习） ----
        # 单独的投影头用于对比学习，可以防止对比学习污染嵌入
        # projection_head.0.weight   # Linear: 256→256
        # projection_head.0.bias
        # projection_head.2.weight   # Linear: 256→256
        # projection_head.2.bias
        # projection_head 结构是 Linear → ReLU → Linear，其中 ReLU 在索引 1，没有参数。
        self.projection_head = nn.Sequential(
            nn.Linear(model_config.embedding_dim, model_config.embedding_dim),
            nn.ReLU(),
            nn.Linear(model_config.embedding_dim, model_config.embedding_dim),
        )

        self.dropout = nn.Dropout(model_config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """使用 Xavier 均匀初始化 Linear 和 LayerNorm 层, Linear 增益 0.5。"""
        # self.modules() 会获取到所有定义的层
        # nn.ReLU 和 nn.Dropout 无参数，nn.ModuleDict 和 nn.ModuleList 是容器，也无参数
        # HeteroConv 和 SAGEConv 内部的Linear 层会被遍历到
        for m in self.modules(): 
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播。

        参数:
            x_dict: 输入特征，{node_type: (num_nodes, in_channels)}
            edge_index_dict: 边索引，{(src, rel, dst): (2, num_edges)}

        返回:
            Tuple[Dict, Dict, Dict]:
                - x_conv: 最终嵌入，{node_type: (num_nodes, embedding_dim)}
                - soc_logits: SOC 分类 logits，{node_type: (num_nodes, num_socs)}
                - proj: 投影嵌入（用于对比学习），{node_type: (num_nodes, embedding_dim)}
        """
        # ---- 1. 特征投影 ----
        x_proj = {}
        for nt in self.node_types:
            if nt in x_dict:
                x_proj[nt] = self.node_lin[nt](x_dict[nt])

        # ---- 2. 多层异构卷积 ----
        # 复制特征投影信息，使用边和多层异构卷积 对特征投影信息进行丰富
        x_conv = x_proj
        # conv: HeteroConv
        for i, conv in enumerate(self.convs):
            #  消息传递（聚合邻居信息）
            x_conv = conv(x_conv, edge_index_dict)
            # 对非最后一层进行 ReLU 激活, Dropout 正则化, L2 归一化
            if i < len(self.convs) - 1:
                for nt in x_conv: #  Iterate over all node types (soc, hlgt, hlt, pt, llt) and apply the same transform to the features of every type.
                    x_conv[nt] = F.relu(x_conv[nt])
                    x_conv[nt] = self.dropout(x_conv[nt])
                    x_conv[nt] = F.normalize(x_conv[nt], p=2, dim=1)

        #  ---- 3. output normalization ----
        #  Iterate over all node types (soc, hlgt, hlt, pt, llt) and apply the same transform to the features of every type.
        for nt in self.node_types:
            if nt in x_conv:
                x_conv[nt] = self.output_norm(x_conv[nt])

        # ---- 4. SOC 分类 ----
        #  Iterate over all node types (soc, hlgt, hlt, pt, llt) and apply the same transform to the features of every type.
        soc_logits = {
            nt: self.soc_classifier(x_conv[nt])
            for nt in x_conv if nt in self.node_types
        }

        # ---- 5. 投影头 ----
        #  Iterate over all node types (soc, hlgt, hlt, pt, llt) and apply the same transform to the features of every type.
        proj = {
            nt: self.projection_head(x_conv[nt])
            for nt in x_conv if nt in self.node_types
        }

        return x_conv, soc_logits, proj


# =============================================================================
# 5. 基础训练器
# =============================================================================

class HeteroMedDRATrainer:
    """
    全量训练器，包含：
        - SOC 分类损失（主任务）
        - 多层级对比损失（辅助任务）
        - 早停策略基于验证集 SOC 分类准确率
    """
    def __init__(
        self,
        model: HeteroMedDRAEncoder,
        data: HeteroData,
        processor: HeteroMedDRADataProcessor,
        training_config: TrainingConfig,
        contrastive_config: ContrastiveConfig,
    ):
        """
        初始化训练器。

        参数:
            model (HeteroMedDRAEncoder): 编码器模型
            data (HeteroData): 异构图数据
            processor (HeteroMedDRADataProcessor): 数据处理器
            training_config (TrainingConfig): 训练超参数配置
            contrastive_config (ContrastiveConfig): 对比学习配置
        """
        self.model = model.to(DEVICE)
        self.data = data.to(DEVICE)
        self.processor = processor
        self.training_config = training_config
        self.contrastive_config = contrastive_config

        # 图信息
        self.node_types = list(data.node_types)
        # 含 8 种 4 前向 + 4 反向 
        self.edge_types = list(data.edge_types)
        # level_map {"SOC": 0, "HLGT": 1, "HLT": 2, "PT": 3, "LLT": 4}
        # idx_to_level {0:"SOC", 1:"HLGT", 2:"HLT", 3:"PT", 4:"LLT"}
        self.idx_to_level = {v: k for k, v in processor.level_map.items()}
        
        # SOC层节点的数量
        self.num_socs = len(processor.level_nodes.get("SOC", []))
        # SOC层所有节点
        self.soc_list = processor.level_nodes.get("SOC", [])
        # {SOC_CODE: index}
        self.soc_to_idx = {soc: i for i, soc in enumerate(self.soc_list)}

        # 准备训练数据
        self.all_nodes: List[int] = []
        self.all_levels: List[int] = []
        self.all_node_types: List[str] = []
        self.all_soc_labels: List[int] = []
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []   # JBI 投稿：独立测试集（仅最终评估一次）
        self.num_train_nodes = 0
        self._prepare_training_data()


        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            # 【新增】评估指标
            "val_macro_f1": [],
            "val_weighted_f1": [],
            "val_macro_precision": [],
            "val_macro_recall": [],
            # 【新增】Per-level 准确率
            "val_hlgt_acc": [],
            "val_hlt_acc": [],
            "val_pt_acc": [],
            "val_llt_acc": [],
            "val_hlgt_macro_f1": [],
            "val_hlt_macro_f1": [],
            "val_pt_macro_f1": [],
            "val_llt_macro_f1": [],
            # 原有
            "train_ce_loss": [],
            "train_cl_loss": [],
            "epoch_times": [],
            "learning_rates": [],
            "intra_soc_sim": [],
            "hierarchical_sim_dict": [],
        }
        # 【新增】最佳模型对应的指标
        self.best_metrics = {}
        # 【JBI 投稿】独立测试集评估指标（训练结束后回填）
        self.test_metrics = {}
        # Sim 诊断计数器（每 50 batch 打印一次）
        self._diag_cnt = 0           # 新增
        self.best_epoch = 0
        self.best_model_state = None
        # 【新增】早停 / 模型选择配置（可在训练前覆盖）
        self.es_primary_metric = "acc"        # 主判据："acc"(默认, 实测信噪比更高) 或 "macro_f1"
        self.es_min_epochs = 60               # 早停保护期：前 N 轮只记录、不累计 patience（≈计划轮数的30%）
        self.es_loss_min_delta = 1e-4         #  Threshold for deciding that the validation loss is "still decreasing"
        self.best_val_loss = float("inf")     #  Best validation loss so far (used for the "still learning" decision)
        self.es_min_delta = 0.001             # 主指标“严格提升”阈值（实测校准：0.002 会在平台期误判无进展）
        self.es_tolerance = 0.005             # 容差带：主指标不落后最优超过该值即视为“同水平”
        self.es_gap_eps = 1e-4                #  Minimum gap improvement required to select a better model inside the tolerance band
        self.collapse_weight = 3              # 塌缩轮等价计为几轮无进展（替代独立的塌缩停止阈值）
        self.collapse_eps = 1e-3              # intra 相对参考值上升的最小量（抑制噪声）
        self.collapse_gap_ratio = 0.1         # gap 相对参考值的最小下降比例（10%）
        self._collapse_ref_intra = None       # 参考 intra（随“选中最佳权重”更新）
        self._collapse_ref_gap = None         # 参考 gap（随“选中最佳权重”更新）
        self.best_val_acc = 0.0               # 历史最高准确率（仅记录/报告用）
        self.best_macro_f1 = 0.0              #  macro_f1 of the selected model
        self.best_primary = 0.0               #  Historical best of the primary criterion
        self.best_gap = 0.0                   #  gap_average of the selected model

    # --------------------- 数据准备 ---------------------

    def _prepare_training_data(self) -> None:
        """按 SOC 分层采样划分训练集和验证集。"""
        logger.info("=" * 60)
        logger.info("准备训练数据（按 SOC 分层采样）")
        logger.info("=" * 60)

        # 遍历所有节点类型
        for nt in self.node_types:
            if hasattr(self.data[nt], "node_codes"):
                # 每层的 node_codes 都是有序的
                for code in self.data[nt].node_codes:
                    level = self.processor.get_node_level(code)
                    # 判断level合法
                    if level and level in self.processor.level_map:
                        if level == "SOC":
                            continue
                        self.all_nodes.append(code)
                        self.all_levels.append(self.processor.level_map[level])
                        self.all_node_types.append(nt)
                        # 获取节点的soc，soc也参与，最终设置为-1
                        soc = self.processor.get_soc_node_code(code)
                        if soc is not None and soc in self.soc_to_idx:
                            # 值为：节点所属于的SOC的索引
                            self.all_soc_labels.append(self.soc_to_idx[soc])
                        else:
                            # 未匹配到 SOC 索引的节点（正常数据不应出现，仅作兜底）
                            self.all_soc_labels.append(-1)

        self.num_train_nodes = len(self.all_nodes)
        logger.info(f"总节点数: {self.num_train_nodes:,}")
        # 遍历每层的节点数
        for nt in self.node_types:
            cnt = sum(1 for t in self.all_node_types if t == nt)
            logger.info(f"    {nt}: {cnt:,}")

 
        # 记录每个SOC中的节点在全局的索引位置
        soc_to_indices = defaultdict(list)
        # soc_lab: soc_index
        for idx, soc_lab in enumerate(self.all_soc_labels):
            # 剔除掉 SOC节点本身
            if soc_lab >= 0:
                soc_to_indices[soc_lab].append(idx)

        # 训练集 / 验证集 / 测试集（JBI 投稿：三分切分，测试集仅最终评估一次）
        train_idx, val_idx, test_idx = [], [], []
        # 设置随机种子
        rng = np.random.RandomState(self.training_config._split_seed)   # 划分种子（见 TrainingConfig）
        _test_ratio = float(getattr(self.training_config, "test_ratio", 0.0) or 0.0)

        # lab：SOC 的索引（0 ~ 26）,
        # indices：属于该 SOC 的所有节点在 all_nodes 中的全局位置列表
        for lab, indices in soc_to_indices.items():
            _lst = list(indices)
            # ---- 先划测试集（仅当 test_ratio > 0；先于验证集，保证测试集独立）----
            if _test_ratio > 0 and len(_lst) >= 3:
                n_test = max(1, min(int(len(_lst) * _test_ratio), len(_lst) // 2))
                test_sel = rng.choice(_lst, n_test, replace=False)
                _test_set = set(int(t) for t in test_sel)
                _rest = [i for i in _lst if i not in _test_set]
            else:
                test_sel, _test_set = [], set()
                _rest = _lst
            # ---- 再在剩余里划验证集 ----
            n_val = max(1, min(int(len(_rest) * self.training_config.val_ratio), len(_rest) // 2))
            # 只有 1 个节点 - 全部用于验证；只有 2 个节点 - 1 验证 1 训练（测试集为空时不改变原二分语义）
            if len(_rest) <= 2:
                val_sel = _rest[:1]           # 取前 1 个元素
                train_sel = _rest[1:]         # 取第 2 个元素到末尾
            else:
                val_sel = rng.choice(_rest, n_val, replace=False)
                _val_set = set(int(v) for v in val_sel)
                train_sel = [i for i in _rest if i not in _val_set]
            test_idx.extend(test_sel)
            val_idx.extend(val_sel)
            train_idx.extend(train_sel)

        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx


        # ---- JBI 投稿：导出/校验「统一划分索引」，保证各脚本用的是同一份划分 ----
        try:
            import json as _sp_json
            _sp_path = os.environ.get("MEDDRA_SPLIT_INDEX_FILE") or os.path.join(
                str(globals().get("CACHE_DIR") or "."), "split_indices.json")
            _sp_payload = {
                "source": os.path.basename(str(globals().get("_SCRIPT_PATH", "") or "")),
                "build_hierarchy_version": str(globals().get("BUILD_HIERARCHY_VERSION", "")),
                "cache_version": str(getattr(GraphCacheManager, "CACHE_VERSION", "")),
                "num_nodes": int(len(self.all_nodes)),
                "split_seed": int(self.training_config._split_seed),
                "val_ratio": float(self.training_config.val_ratio),
                "test_ratio": float(getattr(self.training_config, "test_ratio", 0.0) or 0.0),
                "nodes": [int(c) for c in self.all_nodes],
                "train": [int(i) for i in train_idx],
                "val": [int(i) for i in val_idx],
                "test": [int(i) for i in test_idx],
            }
            if not os.path.exists(_sp_path):
                _sp_dir = os.path.dirname(_sp_path)
                if _sp_dir:
                    os.makedirs(_sp_dir, exist_ok=True)
                with open(_sp_path, "w", encoding="utf-8") as _sp_f:
                    _sp_json.dump(_sp_payload, _sp_f, separators=(",", ":"))
                logger.info(f"\U0001f4cc 划分索引已导出: {_sp_path}"
                            f"（其它脚本设 MEDDRA_SPLIT_INDEX_FILE 指向该文件即可做逐成员校验）")
            else:
                with open(_sp_path, "r", encoding="utf-8") as _sp_f:
                    _sp_old = _sp_json.load(_sp_f)
                for _k in ("num_nodes", "split_seed", "val_ratio", "test_ratio"):
                    if _sp_old.get(_k) != _sp_payload[_k]:
                        raise RuntimeError(
                            f"划分索引与本次不一致（{_k}）：文件={_sp_old.get(_k)}，本次={_sp_payload[_k]}；"
                            f"如需重出划分请另设 MEDDRA_SPLIT_INDEX_FILE 或删除 {_sp_path}")
                if [int(c) for c in _sp_old.get("nodes", [])] != _sp_payload["nodes"]:
                    raise RuntimeError(f"划分索引 {_sp_path} 的节点枚举与本次不一致")
                for _k in ("train", "val", "test"):
                    if set(int(i) for i in _sp_old.get(_k, [])) != set(_sp_payload[_k]):
                        raise RuntimeError(
                            f"划分索引 {_sp_path} 的 {_k} 集与本次不一致——同一协议下各脚本必须共用同一份划分")
                logger.info(f"\U0001f517 划分索引逐成员校验通过: {_sp_path}")
        except RuntimeError:
            raise
        except Exception as _sp_e:
            logger.warning(f"\u26a0\ufe0f 划分索引导出/校验跳过（{_sp_e}）")
        logger.info(f"训练集: {len(train_idx):,} 个节点")
        logger.info(f"验证集: {len(val_idx):,} 个节点")
        logger.info(f"验证集比例: {len(val_idx) / self.num_train_nodes * 100:.1f}%")
        if _test_ratio > 0:
            logger.info(f"测试集: {len(test_idx):,} 个节点（比例 {len(test_idx) / self.num_train_nodes * 100:.1f}%）")

        # 各层训练/验证/测试样本量（分层指标的样本量依据；与 evaluate() 输出的 {level}_n 同口径）
        # 说明：分层变量是 SOC 标签而非层级，各层占比是“同一 SOC 池内四层混合后随机抽样”的自然结果，
        #       故此处显式打印，便于解读分层指标、填写论文表格。SOC 层本就不进入训练/验证/测试集。
        _tr_by_nt, _va_by_nt, _te_by_nt = defaultdict(int), defaultdict(int), defaultdict(int)
        for _i in train_idx:
            _tr_by_nt[self.all_node_types[_i]] += 1
        for _i in val_idx:
            _va_by_nt[self.all_node_types[_i]] += 1
        for _i in test_idx:
            _te_by_nt[self.all_node_types[_i]] += 1
        for _nt in self.node_types:
            _n_lv = _tr_by_nt[_nt] + _va_by_nt[_nt] + _te_by_nt[_nt]
            if _n_lv == 0:
                continue
            logger.info(
                f"    {_nt}: 训练 {_tr_by_nt[_nt]:,} / 验证 {_va_by_nt[_nt]:,}"
                f"（验证占比 {_va_by_nt[_nt] / _n_lv * 100:.1f}%）"
                + (f" / 测试 {_te_by_nt[_nt]:,}" if _test_ratio > 0 else "")
            )

    # --------------------- 辅助查询方法 ---------------------

    def _get_node_index(self, node_code: int, node_type: str) -> Optional[int]:
        """
        获取节点在其类型中的局部索引。
        
        参数:
            node_code (int): 节点代码
            node_type (str): 节点类型小写，如 'hlgt'
            
        返回:
            Optional[int]: 局部索引
        """
        try:
            level = node_type.upper()
            idx = self.processor.level_encoders[level].transform([node_code])[0]
            return int(idx)
        except Exception:
            return None

    # --------------------- 多层级对比损失（向量化版本） ---------------------

    def _compute_multilevel_contrastive_loss(
            self,
            proj_dict: Dict[str, torch.Tensor],
            batch_nodes: List[int],
            batch_types: List[str],
        ) -> torch.Tensor:
        """
        计算多层级对比损失（向量化版本，v7.1：主父/辅父分层）。

        【v7.1 修改说明】
            原先仅使用 parents[0] 作为唯一父节点，导致 PT 层 37.83% 的多父节点
            在对比学习中被错误地当作"非兄弟"处理。现改为：
              - 主父兄弟（强正样本）：共享 PRIMARY_SOC_FG='Y' 确定的主父
              - 辅父兄弟（弱正样本）：仅共享非主父的辅父
              - 父子（基于主父）：用唯一主父确定上下位关系
              - 硬负样本（表兄弟）：共享主父的祖父但主父不同

        正样本：主父兄弟、辅父兄弟、父子
        硬负样本：表兄弟节点（同祖父级但不同主父）

        proj_dict: 前向传播后, 返回的特征信息 全部节点的投影向量
                    通过batch_nodes选中对应的节点计算相似度
        batch_nodes: 批次中所有节点的ID
        batch_types: 批次中所有节点的类型
        """
        # ---- 1. 收集投影嵌入和节点信息 ----
        # 批次所有的节点投影信息
        all_proj = []
        # 当前批次所有节点
        all_codes = []
        # 当前批次所有节点对应的level名称
        all_levels = []

        # 遍历 节点
        for node_code, node_type in zip(batch_nodes, batch_types):
            if node_type not in proj_dict:
                continue
            # 根据node_code获取层级索引
            idx = self._get_node_index(node_code, node_type)
            if idx is None:
                continue
            # 根据层级索引来获取对应的投影值
            all_proj.append(proj_dict[node_type][idx])
            all_codes.append(node_code)
            all_levels.append(self.processor.get_node_level(node_code))

        # 小于两个节点无法计算    
        if len(all_proj) < 2:
            return torch.tensor(0.0, device=DEVICE)

        # 对所有特征进行堆叠 和 L2正则化
        all_proj = F.normalize(torch.stack(all_proj, dim=0), p=2, dim=1)
        N = len(all_codes)

        # ---- 2. 提取父子、祖父信息 ----
        # 【v7.1 关键修改】
        # 主父 = PRIMARY_SOC_FG='Y' 行确定的父（唯一），用于对比学习
        # 所有父 = hierarchy_map 中的所有父节点（含多父），用于辅父兄弟
        # 将 None 替换为 -1，与后续 tensor 计算保持一致

        # ---- 2.1 主父（唯一，用于主父兄弟 / 父子 / 硬负样本）----
        # 所有节点的主父节点
        primary_parents = []
        for n in all_codes:
            # 从 processor 取主父（通过 primary 行确定）
            pp = self.processor._primary_parent.get(n, None)
            primary_parents.append(pp if pp is not None else -1)

        # ---- 2.2 所有父节点（用于辅父兄弟）----
        # 用 padded 矩阵存储多父：[N, MAX_PARENTS]
        # 因为节点的父节点可能不止一个，且数量不一定相同  先定一个最大长度 MAX_PARENTS，不够的用“哨兵值”补齐（padding）
        MAX_PARENTS = 8                    # MedDRA 中 PT 最多 7 个父；换数据版本前须按 diagnose_multi_parent 的 dist 复核
        _max_par = max(len(self.processor._parents_all.get(c, [])) for c in all_codes)
        if _max_par > MAX_PARENTS:
            logger.warning(f"⚠️ 存在 {_max_par} 父节点 > MAX_PARENTS={MAX_PARENTS}，ps[:MAX_PARENTS] 静默截断将丢失主/辅父信息！")
        SENTINEL = -10**9                  # 哨兵值，避免与 -1 或真实 code 冲突
        # 创建一个形状为 [N, MAX_PARENTS] 的张量，所有元素初始化为 SENTINEL
        all_parents_padded = torch.full(
            (N, MAX_PARENTS), SENTINEL, dtype=torch.long, device=DEVICE
        )
        # i 为行索引；  k 为列索引
        for i, code in enumerate(all_codes):
            # 一个节点一行，ps：节点所有的父节点
            ps = self.processor._parents_all.get(code, [])
            for k, p in enumerate(ps[:MAX_PARENTS]):
                all_parents_padded[i, k] = p

        # ---- 2.3 祖父（基于主父的父，用于硬负样本）----
        # 有祖父节点设置为对应的主父节点，没有设置为-1
        grandparents = []
        for n in all_codes:
            pp = self.processor._primary_parent.get(n, None)
            if pp is None:
                grandparents.append(-1)
            else:
                # 主父的父 = 该节点的祖父
                gp = self.processor._primary_parent.get(pp, None)
                grandparents.append(gp if gp is not None else -1)

        # 转换为张量
        primary_parents_t = torch.tensor(primary_parents, dtype=torch.long, device=DEVICE)
        grandparents_t = torch.tensor(grandparents, dtype=torch.long, device=DEVICE)
        # 将level名称（批次所有节点的名称）转化为节点索引
        levels_t = torch.tensor(
            [self.processor.level_map.get(l, -1) for l in all_levels],
            dtype=torch.long,
            device=DEVICE
        )
        codes_t = torch.tensor(all_codes, dtype=torch.long, device=DEVICE)

        # 设置对角矩阵，用于去除自身
        eye_mask = torch.eye(N, dtype=torch.bool, device=DEVICE)

        # ---- 3. 向量化构建掩码 ----
        # [:, None] 将行中的每个元素转化为一行，横向拓展行
        # [None, :] 将所有元素转化为一行，纵向拓展这一行
        # levels_t = [1, 2, 1, 2]  # 形状: (4,)  levels_t[:, None] |levels_t[None, :] — 增加维度
        # levels_t[:, None]        # 形状: (4, 1) 
            # [[1],
            #  [2],
            #  [1],
            #  [2]]
        # levels_t[None, :]        # 形状: (1, 4)
        # [[1, 2, 1, 2]]
        # levels_t[:, None] == levels_t[None, :] 广播
        # [[1, 1, 1, 1],      [[1, 2, 1, 2],
        #  [2, 2, 2, 2],  ==   [1, 2, 1, 2],
        #  [3, 3, 3, 3],       [1, 2, 1, 2],
        #  [4, 4, 4, 4]]       [1, 2, 1, 2]]

        # ---- 3.1 主父有效性掩码 ----
        # 有效 主父节点code = [True, False, True, False] 判断 primary_parents_t 中的值不是-1
        valid_primary_parent = (primary_parents_t != -1)


        c = self.contrastive_config   # 提前取出来，后面都用

        # 主父兄弟的“原始关系”掩码（与开关无关）：同层级 & 同主父 & 双方主父有效 & 非自身。
        # 【C6 语义保障】即使 use_primary_sibling=False，该掩码仍用于第 3.3 节排除这些对子：
        #   否则它们会以“辅父兄弟（权重 0.3）”身份回流，使消融 C6 由“移除主父兄弟”退化为“降权”。
        _primary_sib_raw = (
            (levels_t[:, None] == levels_t[None, :])
            & (primary_parents_t[:, None] == primary_parents_t[None, :])
            & valid_primary_parent[:, None]
            & valid_primary_parent[None, :]
            & ~eye_mask
        )

        # ---- 3.2 主父兄弟掩码（强正样本）----
        if c.use_primary_sibling:
            sibling_mask_primary = _primary_sib_raw
        else:
            # 关闭：该关系对子不再作为正样本（退化为普通负样本），用于消融 C6
            sibling_mask_primary = torch.zeros(
                (N, N), dtype=torch.bool, device=DEVICE
            )


        # ---- 3.3 辅父兄弟掩码（弱正样本）----
        if c.use_secondary_sibling:
            # 排除无效的辅父节点
            valid_all_parents = (all_parents_padded != SENTINEL)
            # 辅父节点
            # all_parents_padded[:, None, :, None] 在维度 1 和维度 3 插入新轴 原形状 [N, M] → 变成 [N, 1, M, 1]
            # all_parents_padded[None, :, None, :] 在维度 0 和维度 2 插入新轴 原形状 [N, M] → 变成 [1, N, 1, M]
            # 比较时的广播 广播规则：从右往左对齐，维度为 1 的轴会自动扩展到匹配。 结果 [N, N, M, M]
            # 两个 节点 只要共享任意一个父节点，就可能是辅父兄弟关系
            # 常配合 .any(dim=(2,3)) 得到 [N, N] 的“是否共享父”矩阵
            shared_any_parent = (
                (all_parents_padded[:, None, :, None] == all_parents_padded[None, :, None, :])
                & valid_all_parents[:, None, :, None]
                & valid_all_parents[None, :, None, :]
            ).any(dim=(2, 3))
            # 同层级 共享辅父 非主父的兄弟节点
            # 注：排除项用与开关无关的 _primary_sib_raw，保证 C6（关主父兄弟）时这些对子
            #     不会被重新计入辅父兄弟（开关开启时二者等价，主实验行为不变）。
            sibling_mask_secondary = (
                shared_any_parent
                & ~_primary_sib_raw
                & (levels_t[:, None] == levels_t[None, :])
                & ~eye_mask
            )
        else:
            sibling_mask_secondary = torch.zeros(
                (N, N), dtype=torch.bool, device=DEVICE
            )



        # ---- 3.4 父子掩码（基于主父）----
        if c.use_parent:
            #  parent-child
            parent_mask = (
                (primary_parents_t[:, None] == codes_t[None, :])
                | (primary_parents_t[None, :] == codes_t[:, None])
            )
            parent_mask = (
                parent_mask
                & valid_primary_parent[:, None]
                & valid_primary_parent[None, :]
            )
        else:
            parent_mask = torch.zeros((N, N), dtype=torch.bool, device=DEVICE)

        # ---- 3.5 硬负样本掩码（基于主父的祖父）----
        if c.use_hard_negative:
            valid_grandparent = (grandparents_t != -1)
            # 同层级 同主父 非主父的祖父节点
            hard_neg_mask = (
                (grandparents_t[:, None] == grandparents_t[None, :])
                & (primary_parents_t[:, None] != primary_parents_t[None, :])
                & (levels_t[:, None] == levels_t[None, :])
                & valid_grandparent[:, None]
                & valid_grandparent[None, :]
                & valid_primary_parent[:, None]
                & valid_primary_parent[None, :]
                & ~eye_mask
            )
        else:
            hard_neg_mask = torch.zeros((N, N), dtype=torch.bool, device=DEVICE)


        # ---- 3.6 正样本与硬负样本互斥 ----
        # 正样本
        is_positive_mask = (
            sibling_mask_primary
            | sibling_mask_secondary
            | parent_mask
        )
        # 正负互斥：负样本 = 负样本 与 正样本的互斥
        if c.use_mutual_exclusion:
            hard_neg_mask = hard_neg_mask & ~is_positive_mask



        # ---- 4. 计算 InfoNCE 损失 ----

        # 相似度矩阵
        sim_matrix = all_proj @ all_proj.T / c.temperature
        exp_sim = torch.exp(sim_matrix)

        # 1. 计算正样本权重矩阵（不再转为 bool）
        # 【v7.1 修改】三种正样本：主父兄弟（强）、辅父兄弟（弱）、父子
        positive_weights = (
            sibling_mask_primary.float()    * c.sibling_weight
            + sibling_mask_secondary.float() * c.sibling_secondary_weight
            + parent_mask.float()            * c.parent_weight
        )

        # 2. 分子：直接使用权重矩阵加权求和
        positive_sum = (exp_sim * positive_weights).sum(dim=1)

        # 3. 判断哪些是正样本（用于排除正样本的掩码，权重 > 0 即为正）
        is_positive = positive_weights > 0

        # 4. 简单负样本掩码：排除正样本和自己
        simple_neg_mask = ~(is_positive | eye_mask)

        # 5. 负样本权重
        neg_weights = simple_neg_mask.float() + hard_neg_mask.float() * (c.hard_negative_weight - 1.0)

        # 6. 分母计算
        all_sum = positive_sum + (exp_sim * neg_weights).sum(dim=1) + 1e-8

        # 7. 计算损失
        # 取负对数，值域 (0, ∞)。比例越高，损失越小
        loss = -torch.log((positive_sum / all_sum).clamp(min=1e-8))
        # 只对有正样本的锚点计算损失（避免无效节点影响）
        loss = loss[positive_sum > 0].mean()

        # ---- 5. Sim 诊断（每 50 个 batch 打印一次） ----
        # 【v7.1 新增】用于观察正/负样本对的相似度分布，指导 temperature 调整
        self._diag_cnt += 1

        if self._diag_cnt % 50 == 0:
            with torch.no_grad():
                sim_raw = all_proj @ all_proj.T       # 未除温度
                off_diag = ~eye_mask  # 去除自身
                sim_off = sim_raw[off_diag]

                # 复用 3.6 节定义的 is_positive_mask，避免重复计算与语义漂移
                is_pos = is_positive_mask[off_diag]
                is_hard = hard_neg_mask[off_diag]
                is_sneg = simple_neg_mask[off_diag]

                pos_mean = sim_off[is_pos].mean().item() if is_pos.any() else 0.0
                hard_mean = sim_off[is_hard].mean().item() if is_hard.any() else 0.0
                sneg_mean = sim_off[is_sneg].mean().item() if is_sneg.any() else 0.0
                gap = pos_mean - sneg_mean

                logger.info(
                    f"  [Sim诊断] 全部:[{sim_off.min():.3f},{sim_off.max():.3f}] "
                    f"mean={sim_off.mean():.3f} | "
                    f"pos={pos_mean:.3f} hard={hard_mean:.3f} sneg={sneg_mean:.3f} | "
                    f"Δ={gap:.3f} logit_gap={gap / c.temperature:.2f} | "
                    f"mask密度: "
                    f"sib1={sibling_mask_primary.float().mean():.4f} "
                    f"sib2={sibling_mask_secondary.float().mean():.4f} "
                    f"par={parent_mask.float().mean():.4f} "
                    f"hard={hard_neg_mask.float().mean():.4f}"
                )

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=DEVICE)
    
    
    # --------------------- 采样估计 inter / 表兄弟硬负相似度 ---------------------
    def _sampled_inter_and_cousin_similarity(
        self,
        nodes: List[int],
        emb_dict: Dict[int, np.ndarray],
        n_sample_nodes: int = 20000,
        n_pairs: int = 20000,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """
        用蒙特卡洛采样估计某一层级上的两个“对照相似度”：

            - inter: 两个节点在结构上“无任何共同父节点”（即分属不同簇）时的平均余弦相似度；
                     代表该层级的跨簇语义距离，越低越好。
            - cousin: 表兄弟硬负样本对（共享祖父、主父不同）的平均余弦相似度；
                     对应训练时对比损失的 hard_negative 定义，理想值应明显低于 intra。

        设计说明：
            1. 固定随机种子，保证逐 epoch 采样同一批节点对，便于观察趋势；
            2. 采样节点数上限 n_sample_nodes，避免 LLT(约 9 万节点) 全量两两比较导致 OOM；
            3. 缺少主父信息时 cousin 返回 0.0（例如旧缓存未保存 primary_parent）。

        参数:
            nodes (List[int]): 该层级全部节点代码
            emb_dict (Dict[int, np.ndarray]): {node_code: embedding}
            n_sample_nodes (int): 参与采样的节点数上限
            n_pairs (int): 采样的节点对数
            seed (int): 采样随机种子

        返回:
            Tuple[float, float]: (inter 平均余弦, cousin 平均余弦)
        """
        valid = [n for n in nodes if n in emb_dict]
        if len(valid) < 2:
            return 0.0, 0.0

        rng = np.random.RandomState(seed)

        # 限制参与采样的节点规模，控制显存与耗时
        if len(valid) > n_sample_nodes:
            sel = rng.choice(len(valid), n_sample_nodes, replace=False)
            # 取出采用的节点
            valid = [valid[i] for i in sel]

        # 节点总数
        M = len(valid)
        if M < 2:
            return 0.0, 0.0

        # L2 归一化后，点积即为余弦相似度
        E = np.vstack([emb_dict[n] for n in valid]).astype(np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

        # 结构信息：父集合，以及祖父（主父的主父）
        parent_sets = [
            set(self.processor.hierarchy_map.get(n, {}).get("parent", []))
            for n in valid
        ]
        primary_parent_map = getattr(self.processor, "_primary_parent", {}) or {}
        primary_parent = [primary_parent_map.get(n, None) for n in valid]
        grandparent = [
            primary_parent_map.get(pp, None) if pp is not None else None
            for pp in primary_parent
        ]

        # 采样节点对并去除自身配对
        # 随机生成一批索引，范围是 [0, M)，数量是 n_pairs 个
        i = rng.randint(0, M, n_pairs)
        j = rng.randint(0, M, n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return 0.0, 0.0

        # np.einsum("ij,ij->i", A, B)：对每一行做逐元素相乘再求和，也就是行内点积
        # ij	第一个输入 E[i] 的下标：行 i，列 j
        # ij	第二个输入 E[j] 的下标：行 i，列 j
        # ->i	输出只保留行下标 i，列下标 j 被求和消掉
        # 等价写法 cos = (E[i] * E[j]).sum(axis=1)
        cos = np.einsum("ij,ij->i", E[i], E[j])
        

        # inter：两个节点无任何共同父节点
        # np.fromiter 转成 NumPy 布尔数组
        no_shared = np.fromiter(
            (len(parent_sets[a] & parent_sets[b]) == 0 for a, b in zip(i, j)),
            dtype=bool, count=len(i),
        )

        # 得到布尔数组
        # cousin：共享祖父（非空）且主父不同
        # 同祖父（主），不同父（主）
        cousin = np.fromiter(
            (
                grandparent[a] is not None
                and grandparent[a] == grandparent[b]
                and primary_parent[a] != primary_parent[b]
                for a, b in zip(i, j)
            ),
            dtype=bool, count=len(i),
        )
        
        # 计算不同关系的平均余弦相似度
        # cos[no_shared] 根据布尔数组取出对应的节点
        inter = float(cos[no_shared].mean()) if no_shared.any() else 0.0
        cousin_sim = float(cos[cousin].mean()) if cousin.any() else 0.0
        return inter, cousin_sim

    def compute_hierarchical_intra_similarity(
        self, 
        embeddings_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # embeddings_dict 全部节点的嵌入矩阵
        """
        计算所有层级的簇内平均余弦相似度（优化版）。
        
        优化点：
            1. 使用有效节点数（去重）进行加权平均
            2. 避免 PT 节点重复计数导致权重失真
        
        返回:
            Dict[str, float]: {
                'HLGT': float,   # 同一 SOC 下的 HLGT 平均余弦相似度
                'HLT': float,    # 同一 HLGT 下的 HLT 平均余弦相似度
                'PT': float,     # 同一 HLT 下的 PT 平均余弦相似度
                'LLT': float,    # 同一 PT 下的 LLT 平均余弦相似度
                'average': float # 按有效节点数量加权的平均相似度
            }
        """
        results = {}
        
        # ---- 1. 预缓存节点索引和嵌入 ----
        # {node_type:{node_code: node_index....} ...}
        node_index_cache: Dict[str, Dict[int, int]] = {}
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            ntype = level.lower()
            # 获取当前层次中所有的节点
            nodes = self.processor.get_nodes_by_level(level)
            index_cache = {}
            for node in nodes:
                # 根据节点 和 Level（node_type）获取索引（Level层的索引）
                idx = self._get_node_index(node, ntype)
                if idx is not None:
                    index_cache[node] = idx
            node_index_cache[ntype] = index_cache

        # {node_type: {node_code: node_embedding...}}
        emb_cache: Dict[str, Dict[int, np.ndarray]] = {}
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            ntype = level.lower()
            if ntype not in embeddings_dict:
                emb_cache[ntype] = {}
                continue
            emb_tensor = embeddings_dict[ntype]
            index_cache = node_index_cache.get(ntype, {})
            emb_cache[ntype] = {
                # 根据索引获取嵌入向量 node_index_cache 的index 与 特征矩阵（嵌入矩阵）使用同一套
                node: emb_tensor[idx].cpu().numpy()
                for node, idx in index_cache.items()
            }
        
        # ---- 2. 定义层级配置 ---- 子 - 父
        level_configs = [
            ('HLGT', 'SOC'),   # HLGT: 按 SOC 分组
            ('HLT', 'HLGT'),   # HLT: 按 HLGT 分组
            ('PT', 'HLT'),     # PT: 按 HLT 分组（多父节点支持）
            ('LLT', 'PT'),     # LLT: 按 PT 分组
        ]
        
        # ---- 3. 存储有效节点数（去重） ----
        # {Level:node_count}
        effective_counts = {}
        
        for current_level, parent_level in level_configs:
            ntype = current_level.lower()
            #  all nodes at the current level
            nodes = self.processor.get_nodes_by_level(current_level)

            # 小于2无法计算相似度
            if len(nodes) < 2:
                results[current_level] = 0.0
                effective_counts[current_level] = 0
                continue
            
            # 按目标层级分组  将一个父节点中的所有节点的嵌入矩阵放在一个list中，两两计算余弦相似度
            group_to_embs = defaultdict(list)
            # 当前层所有节点的嵌入向量
            emb_dict = emb_cache.get(ntype, {})
            
            # 记录实际参与计算的节点（去重）
            unique_nodes = set()
            
            for node in nodes:
                # 获取嵌入矩阵
                emb = emb_dict.get(node)
                if emb is None:
                    continue
                
                # 直接获取所有父节点
                parent_nodes = self.processor.hierarchy_map.get(node, {}).get("parent", [])
                
                # 如果该节点有父节点，记录它参与了计算
                if parent_nodes:
                    unique_nodes.add(node)

                # 主要处理一个节点有多个父节点
                for parent_node in parent_nodes:
                    if parent_node is not None:
                        # 向父节点中添加嵌入向量
                        group_to_embs[parent_node].append(emb)
            
            # 存储有效节点数
            effective_counts[current_level] = len(unique_nodes)
            
            # 计算每个组内的平均相似度 - 当前层
            sims = []
            for embs in group_to_embs.values():
                # 小于2直接跳过
                if len(embs) < 2:
                    continue
                # 堆叠
                embs = np.vstack(embs)
                # 计算所有节点两两之间的余弦相似度
                sim_mat = cosine_similarity(embs)

                # np.triu_indices_from(): 提取上三角矩阵（Upper Triangle） 的索引，
                # 且排除对角线 k=1: 从主对角线往上数第 1 条对角线开始，即不包含对角线
                #   相似度矩阵是对称的（sim_mat[i][j] == sim_mat[j][i]），并且对角线上的值永远是 1.0（自己和自己相似）。
                triu = np.triu_indices_from(sim_mat, k=1)
                # 安全性校验
                # triu[0] 是行索引数组，triu[0].size 就是上三角元素的个数。
                if triu[0].size > 0:
                    # sim_mat[triu]: 取这些上三角位置对应的相似度值
                    # np.mean()计算它们的平均值 - 每个组
                    sims.append(np.mean(sim_mat[triu]))

            # 层级 相似度值 - 均值
            results[current_level] = np.mean(sims) if sims else 0.0


        
        # ---- 4. 计算加权平均（使用有效节点数） ----
        total_nodes = 0
        weighted_sum = 0.0
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            # 优先使用有效节点数，如果为 0 则回退到原始总数（兼容性）
            n_nodes = effective_counts.get(level, self.processor.num_nodes_by_level.get(level, 0))
            total_nodes += n_nodes
            # 层级权重 = 均值 * 参与计算节点数
            weighted_sum += results.get(level, 0) * n_nodes
        # 平均值 = 总层级权重 / 参与计算总节点数
        results['average'] = weighted_sum / total_nodes if total_nodes > 0 else 0.0

        # ---- 4b. 计算均匀加权平均 ----
        uniform_sum = sum(results.get(level, 0) for level in ['HLGT', 'HLT', 'PT', 'LLT'])
        results['uniform_average'] = uniform_sum / 4.0

        # ---- 5. 采样估计 inter（跨簇）与 cousin（表兄弟硬负）相似度，并计算 gap ----
        # gap = intra - inter：越大说明“组内近、组间远”，判别性越强；
        # 若发生表示塌缩，intra 与 inter 会同时升高，gap 反而缩小，可作为塌缩预警。
        inter_res: Dict[str, float] = {}
        cousin_res: Dict[str, float] = {}
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            ntype = level.lower()
            # 跨簇      表兄弟硬负
            inter_val, cousin_val = self._sampled_inter_and_cousin_similarity(
                self.processor.get_nodes_by_level(level),
                emb_cache.get(ntype, {}),
            )
            # 跨簇
            inter_res[level] = inter_val
            # 表兄弟硬负
            cousin_res[level] = cousin_val
            # 正样本层级相似度
            intra_val = float(results.get(level, 0.0))
            results[f"inter_{level}"] = inter_val
            results[f"cousin_{level}"] = cousin_val
            # 正样本 - 跨簇
            results[f"gap_{level}"] = intra_val - inter_val
            # 正样本 - 表兄弟硬负
            results[f"gap_cousin_{level}"] = intra_val - cousin_val

        # 加权平均（与 intra 的加权口径一致，按有效节点数）
        def _weighted_avg(per_level: Dict[str, float]) -> float:
            tot, wsum = 0.0, 0.0
            for _lv in ['HLGT', 'HLT', 'PT', 'LLT']:
                _n = effective_counts.get(_lv, self.processor.num_nodes_by_level.get(_lv, 0))
                tot += _n
                wsum += per_level.get(_lv, 0.0) * _n
            return wsum / tot if tot > 0 else 0.0

        results['inter_average'] = _weighted_avg(inter_res)
        results['cousin_average'] = _weighted_avg(cousin_res)
        results['gap_average'] = float(results.get('average', 0.0)) - results['inter_average']
        results['gap_cousin_average'] = float(results.get('average', 0.0)) - results['cousin_average']

        # 跨簇 和 表兄弟硬负 全局均值
        _inter_uniform = float(np.mean([inter_res[_lv] for _lv in ['HLGT', 'HLT', 'PT', 'LLT']]))
        results['inter_uniform_average'] = _inter_uniform
        results['gap_uniform_average'] = float(results.get('uniform_average', 0.0)) - _inter_uniform
        
        # 总计数信息
        results['effective_counts'] = effective_counts
        
        return results

    # --------------------- 断点续训 ---------------------
    # 设计要点：
    #   1) 落盘"再跑一轮所需的一切"：模型 / 优化器 / 调度器 / 最佳权重 / 早停计数 / history / RNG；
    #   2) 指纹一致性：配置、协议参数、构建版本、图规模任一变化 → 拒绝续训（防止一次实验混两套设置）；
    #   3) 数值可复现性：训练本身按 `train_seed*1000 + epoch` 逐轮播种（train_seed 未显式设置时
    #      等于 random_state；见 train_epoch 与 TrainingConfig 的多 seed 方差估计说明），
    #      不依赖全局 RNG 态，故续训后的轨迹与"不中断训练"逐位一致；RNG 态仍一并保存，只为让训练
    #      结束后的绘图采样也可复现；
    #   4) 落盘用"临时名 + os.replace"原子替换，避免写到一半被杀导致断点文件损坏；
    #   5) 只含原生类型与张量（numpy 标量经 _resume_sanitize 转换），兼容 torch>=2.6 的
    #      torch.load(weights_only=True) 默认行为。

    def _resume_fingerprint(self, cfg: TrainingConfig) -> Dict:
        """续训一致性指纹（全原生类型）。任一字段不同即判定"不是同一次实验"。"""
        from dataclasses import fields as _fields
        return {
            "build_hierarchy_version": BUILD_HIERARCHY_VERSION,
            "cache_version": GraphCacheManager.CACHE_VERSION,   # 类属性（单一来源见模块顶部）
            # resume_file 是"运维选项"（决定从哪读断点），不影响实验可比性，故**不进指纹**：
            # 这样它既不会让既有断点失效（旧断点的 training 字段 = 全部旧字段，与本式结果逐字一致），
            # 也不会因为"换了个断点来源"而被误判成另一次实验。
            "training": {f.name: getattr(cfg, f.name) for f in _fields(cfg)
                         if f.name != "resume_file"},
            "contrastive": {f.name: getattr(self.contrastive_config, f.name)
                            for f in _fields(self.contrastive_config)},
            "num_train_nodes": int(self.num_train_nodes),
            "num_soc_classes": int(self.num_socs),
            "node_types": [str(t) for t in self.node_types],
            "model_class": type(self.model).__name__,
            "model_params": int(sum(p.numel() for p in self.model.parameters())),
        }

    def _save_resume(self, cfg: TrainingConfig, optimizer, scheduler, epoch: int,
                     patience_counter: int, collapse_counter: int,
                     collapse_weight_total: int) -> None:
        """落盘续训文件。调用点保证"本轮全部状态已更新完毕"（保存后从 epoch+1 继续）。"""
        _ns, _narr, _npos, _nhasg, _ncached = np.random.get_state()
        # 整包再过一次 _resume_sanitize：兜住任何漏网的 numpy 标量（torch>=2.6 的
        # weights_only=True 只接受原生类型与张量，numpy 标量会让断点读不回来）
        payload = _resume_sanitize({
            "fingerprint": self._resume_fingerprint(cfg),
            "epoch": int(epoch),                      # 已完成的轮号（0 基）
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": _resume_sanitize(self.history),
            "best": {
                "val_acc": float(self.best_val_acc),
                "primary": float(self.best_primary),
                "gap": float(self.best_gap),
                "epoch": int(self.best_epoch),
                "macro_f1": float(self.best_macro_f1),
                "val_loss": float(self.best_val_loss),
                "metrics": _resume_sanitize(self.best_metrics),
                "model_state": self.best_model_state,   # 全为 CPU 张量
                "ref_intra": (None if self._collapse_ref_intra is None
                              else float(self._collapse_ref_intra)),
                "ref_gap": (None if self._collapse_ref_gap is None
                            else float(self._collapse_ref_gap)),
            },
            "counters": {
                "patience": int(patience_counter),
                "collapse": int(collapse_counter),
                "collapse_weight_total": int(collapse_weight_total),
            },
            "rng": {
                "python": random.getstate(),
                "numpy": [str(_ns), _narr.tolist(), int(_npos), int(_nhasg), float(_ncached)],
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        })
        _, _resume_write = _resume_paths(cfg)   # 指定文件只用于读取，写入始终用默认路径
        _tmp = _resume_write + ".tmp"
        try:
            torch.save(payload, _tmp)
            os.replace(_tmp, _resume_write)
            logger.info(f"💾 断点已保存（已完成 epoch {epoch + 1}）: {_resume_write}")
        except Exception as e:
            logger.warning(f"⚠️ 断点保存失败（不影响训练继续）: {e}")

    def _load_resume(self, cfg: TrainingConfig, optimizer, scheduler):
        """读取续训文件并原地恢复状态。

        返回 (下一轮起始 epoch 号, patience, collapse_counter, collapse_weight_total)；
        文件不存在/损坏/指纹不一致/恢复失败时返回 None（调用方按"从头训练"处理）。
        """
        try:
            _resume_read, _ = _resume_paths(cfg)
            if _resume_read != RESUME_FILE:
                logger.info(f"📂 本次从配置指定的断点文件读取: {_resume_read}")
            st = torch.load(_resume_read, map_location="cpu")
        except Exception as e:
            logger.warning(f"⚠️ 断点文件读取失败，忽略并从头训练: {e}")
            return None

        if not isinstance(st, dict) or st.get("fingerprint") != self._resume_fingerprint(cfg):
            logger.warning(
                "⚠️ 断点的配置/版本/图规模与本次不一致 → 忽略断点、从头训练"
                "（该检查用于防止一次实验混入两套设置）"
            )
            return None

        try:
            self.model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            self.history = st["history"]
            _b = st["best"]
            self.best_val_acc = float(_b["val_acc"])
            self.best_primary = float(_b["primary"])
            self.best_gap = float(_b["gap"])
            self.best_epoch = int(_b["epoch"])
            self.best_macro_f1 = float(_b["macro_f1"])
            self.best_val_loss = float(_b["val_loss"])
            self.best_metrics = _b["metrics"]
            self.best_model_state = _b["model_state"]
            self._collapse_ref_intra = _b["ref_intra"]
            self._collapse_ref_gap = _b["ref_gap"]

            _rng = st.get("rng") or {}
            if "python" in _rng:
                random.setstate(_rng["python"])
            if "numpy" in _rng:
                _s, _a, _p, _h, _c = _rng["numpy"]
                np.random.set_state((_s, np.array(_a, dtype=np.uint32), _p, _h, _c))
            if "torch" in _rng:
                torch.set_rng_state(_rng["torch"].to(torch.uint8))
            if _rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.to(torch.uint8) for s in _rng["cuda"]])
        except Exception as e:
            logger.warning(f"⚠️ 断点状态恢复失败，忽略并从头训练: {e}")
            return None

        _c = st["counters"]
        return (int(st["epoch"]) + 1, int(_c["patience"]),
                int(_c["collapse"]), int(_c["collapse_weight_total"]))


    # --------------------- 训练 Epoch ---------------------

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0
    ) -> Tuple[float, float, float, float]:
        """
        训练一个 epoch。

        【v7.1 性能修复】
            原先每个 batch 都调用一次 self.model(x_dict, edge_dict)，而 x_dict/edge_dict
            是整张图（12 万节点）。每个 epoch 有 ~211 个 batch，200 epoch 总计 ~42,200 次
            全图前向传播，训练极慢。
            现改为：每个 epoch 只做一次全图前向传播，所有 batch 复用 x_out / soc_logits / proj，
            累积所有 batch 的 loss 后一次性 backward + step。
            语义变化：从 mini-batch SGD 变为 full-batch GD（每 epoch 更新一次权重），
            对 12 万节点的图通常收敛更稳，但可能需要适当调大 lr 或 epochs。

        参数:
            optimizer (torch.optim.Optimizer): 优化器

        返回:
            Tuple[float, float, float, float]: (平均总损失, 准确率, CE损失, CL损失)
        """
        # 0 训练模式
        self.model.train()

        cfg = self.training_config
        total_loss, total_ce, total_cl = 0.0, 0.0, 0.0
        correct, total = 0, 0
        num_batches = 0

        if len(self.train_indices) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # 对索引进行随机打乱（逐轮播种使用训练种子 train_seed，与全局初始化同源；
        # train_seed 未显式设置时回落 random_state，与历史运行逐位一致）
        rng = np.random.RandomState(self.training_config._train_seed * 1000 + epoch)
        indices = rng.permutation(self.train_indices)

        # 构建特征字典
        x_dict = {
            # 根据节点类型或者data中的特征
            nt: self.data[nt].x
            for nt in self.node_types if hasattr(self.data[nt], "x")
        }
        # 构建边字典
        edge_dict = {
            # 根据边类型获取节点中的边索引
            et: self.data[et].edge_index
            for et in self.edge_types if hasattr(self.data[et], "edge_index")
        }

        # ★ 性能关键：每个 epoch 只做一次全图前向传播
        x_out, soc_logits, proj = self.model(x_dict, edge_dict)

        # 收集所有 batch 的 loss，最后一次性 backward
        batch_losses: List[torch.Tensor] = []

        # 遍历训练集，以 batch_size 为步长
        for i in range(0, len(indices), cfg.batch_size):
            # 批次索引
            batch_idx = indices[i:i + cfg.batch_size]

            # 批次节点
            batch_nodes = [self.all_nodes[idx] for idx in batch_idx]
            # 批次soc，即 target  all_soc_labels 值为soc的索引(0~26)
            batch_socs = [self.all_soc_labels[idx] for idx in batch_idx]
            # 批次Level
            batch_types = [self.all_node_types[idx] for idx in batch_idx]

            # ---- SOC 分类损失 ----
            ce_loss = 0.0
            ntypes_used = 0
            batch_correct, batch_total = 0, 0

            # list 按照 batch_nodes 的顺序存储
            # 每个Level下所有的节点
            by_type = defaultdict(list)
            # 每个Level下所有节点的soc(索引)
            soc_by_type = defaultdict(list)
            for n, s, t in zip(batch_nodes, batch_socs, batch_types):
                by_type[t].append(n)
                soc_by_type[t].append(s)

            # t: 层级
            # nodes: 批次中，层级所有节点
            for t, nodes in by_type.items():
                if t not in soc_logits:
                    continue
                # 当前层的分类预测信息
                logits = soc_logits[t]
                # 转换为层级索引  idxs 的顺序与nodes一致
                idxs = [self._get_node_index(n, t) for n in nodes]
                valid_idxs = [i for i in idxs if i is not None]
                if not valid_idxs:
                    continue
                # 有效预测信息 
                valid_logits = logits[valid_idxs]
                # 真实信息
                valid_socs = torch.tensor(
                    [soc_by_type[t][j] for j, idx in enumerate(idxs) if idx is not None],
                    dtype=torch.long,
                    device=DEVICE,
                )
                if len(valid_socs) > 0:
                    # 3. 计算损失 - 交叉熵
                    loss_ce = F.cross_entropy(valid_logits, valid_socs)
                    # 将该批次节点类型的损失累加到总损失中
                    ce_loss += loss_ce
                    # 记录参与损失的节点类型数量
                    ntypes_used += 1
                    # 模型预测概率最高的 SOC 类别
                    pred = valid_logits.argmax(dim=1)
                    # 统计预测正确的样本数
                    batch_correct += (pred == valid_socs).sum().item()
                    # 统计该批次中参与计算的节点总数
                    batch_total += len(valid_socs)

            if batch_total > 0 and ntypes_used > 0:
                # 平均损失 = (类型1损失 + 类型2损失 + ...) / 参与损失计算的类型数量
                ce_loss = ce_loss / ntypes_used
            else:
                ce_loss = torch.tensor(0.0, device=DEVICE)

            # ---- 对比损失 ----
            cl_loss = torch.tensor(0.0, device=DEVICE)
            # 计算多层级对比损失
            if cfg.use_contrastive and proj:
                cl_loss = self._compute_multilevel_contrastive_loss(proj, batch_nodes, batch_types)

            # ---- 总损失 ----
            # 总损失 = 分类损失 + 多层级对比损失权重 * 多层级对比损失
            loss = ce_loss + self.contrastive_config.contrastive_weight * cl_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("检测到 NaN/Inf 损失，跳过此批次")
                continue

            # 收集该 batch 的 loss（不立即 backward）
            batch_losses.append(loss)

            total_loss += loss.item()
            total_ce += ce_loss.item() if ce_loss > 0 else 0
            total_cl += cl_loss.item() if cl_loss > 0 else 0
            correct += batch_correct
            total += batch_total
            num_batches += 1

        # ★ 所有 batch 处理完后，一次性 backward + step
        if batch_losses:
            # 清空的是上一轮迭代残留的梯度
            optimizer.zero_grad()
            total_batch_loss = torch.stack(batch_losses).mean()
            # 4. 后向传播
            total_batch_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            # 5. 梯度更新
            optimizer.step()

        if num_batches == 0:
            return 0.0, 0.0, 0.0, 0.0

        avg_loss = total_loss / num_batches
        avg_acc = correct / max(1, total)
        avg_ce = total_ce / num_batches
        avg_cl = total_cl / num_batches
        return avg_loss, avg_acc, avg_ce, avg_cl



    # --------------------- 结构线索诊断（方案①） ---------------------

    _DIAG_LEVELS = ("hlt", "pt", "llt")

    def _get_train_code_set(self) -> set:
        """
        被监督（训练集）节点的 code 集合，用于结构线索诊断。

        训练/验证划分在训练开始前已固定且不随 epoch 变化，因此惰性构建一次并缓存。
        """
        if getattr(self, "_train_code_set", None) is None:
            self._train_code_set = {self.all_nodes[i] for i in self.train_indices}
        return self._train_code_set

    def _parent_supervised(self, node_code: int, ntype: str) -> Optional[bool]:
        """
        诊断用：判断节点的主父（由 PRIMARY_SOC_FG='Y' 确定）是否在被监督的训练集内。

        返回 True / False；返回 None 表示该节点不参与诊断：
          · HLGT 的主父是 SOC，而 SOC 层不携带标签、不参与训练，该诊断对其无意义；
          · 主父信息缺失（旧缓存未保存 primary_parent）。

        用途：把验证集拆成“主父∈train”与“主父∉train”两个子群比较准确率。由于图构建时
        为每条父子边生成了反向边（消息可由子流向父），被监督的父节点会直接为子节点提供
        标签线索；两子群的准确率差可作为该“结构线索”贡献的经验上界。该指标仅用于分析，
        不参与早停与模型选择。
        """
        if ntype not in self._DIAG_LEVELS:
            return None
        pp = self.processor._primary_parent.get(node_code, None)
        if pp is None:
            return None
        return pp in self._get_train_code_set()



    # --------------------- 验证 ---------------------

    def evaluate(self, indices: Optional[List[int]] = None) -> Tuple[float, float, float, Dict, Dict]:
        """
        在指定索引集上评估模型（默认验证集；传入 test_indices 即评估测试集）。
        
        参数:
            indices (Optional[List[int]]): 要评估的节点索引列表；None 时默认 self.val_indices。
        
        返回:
            Tuple[float, float, float, Dict, Dict]:
                - val_loss
                - val_acc
                - intra_sim_avg
                - hierarchical_sim_dict
                - metrics: 完整指标字典
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            confusion_matrix,
        )
        
        _eval_indices = self.val_indices if indices is None else indices

        # 1. 模型切换为验证模式
        self.model.eval()
        if len(_eval_indices) == 0:
            return 0.0, 0.0, 0.0, {}, {}
        
        x_dict = {nt: self.data[nt].x for nt in self.node_types if hasattr(self.data[nt], "x")}
        edge_dict = {et: self.data[et].edge_index for et in self.edge_types if hasattr(self.data[et], "edge_index")}
        
        # 2. 前向传播获取输出
        with torch.no_grad():
            x_out, soc_logits, _ = self.model(x_dict, edge_dict)
        
        total_loss = 0.0
        correct, total = 0, 0
        num_batches = 0
        
        # 【新增】收集所有预测和真实标签
        all_preds, all_truths = [], []
        per_level_preds = defaultdict(list)
        per_level_truths = defaultdict(list)

        # 【诊断·方案①】结构线索通道计数器：按“主父是否在训练集内”分组
        # in 在  out 不在  lv 按照层级划分
        diag_correct = {"in": 0, "out": 0}
        diag_total = {"in": 0, "out": 0}
        diag_correct_lv = {lv: {"in": 0, "out": 0} for lv in self._DIAG_LEVELS}
        diag_total_lv = {lv: {"in": 0, "out": 0} for lv in self._DIAG_LEVELS}
        
        batch_size = self.training_config.batch_size
        for i in range(0, len(_eval_indices), batch_size):
            batch_idx = _eval_indices[i:i+batch_size]
            batch_nodes = [self.all_nodes[idx] for idx in batch_idx]
            batch_socs = [self.all_soc_labels[idx] for idx in batch_idx]
            batch_types = [self.all_node_types[idx] for idx in batch_idx]
            
            by_type = defaultdict(list)
            soc_by_type = defaultdict(list)
            for node, soc, ntype in zip(batch_nodes, batch_socs, batch_types):
                by_type[ntype].append(node)
                soc_by_type[ntype].append(soc)
            
            for ntype, nodes in by_type.items():
                if ntype not in soc_logits:
                    continue
                logits = soc_logits[ntype]
                # 节点索引
                idxs = [self._get_node_index(n, ntype) for n in nodes]
                valid_idxs = [i for i in idxs if i is not None]
                if not valid_idxs:
                    continue
                # 真实预测值
                valid_logits = logits[valid_idxs]
                # 真实标签值
                valid_socs = torch.tensor(
                    [soc_by_type[ntype][j] for j, idx in enumerate(idxs) if idx is not None],
                    dtype=torch.long, device=DEVICE,
                )
                if len(valid_socs) > 0:
                    # 计算CE损失
                    loss = F.cross_entropy(valid_logits, valid_socs)
                    total_loss += loss.item()
                    num_batches += 1
                    # 预测值
                    pred = valid_logits.argmax(dim=1)
                    # 正确数/总数
                    correct += (pred == valid_socs).sum().item()
                    total += len(valid_socs)
                    
                    # 【新增】收集预测
                    pred_list = pred.cpu().tolist()
                    truth_list = valid_socs.cpu().tolist()
                    all_preds.extend(pred_list)
                    all_truths.extend(truth_list)
                    per_level_preds[ntype].extend(pred_list)
                    per_level_truths[ntype].extend(truth_list)

                    # 【诊断·方案①】按节点粒度累计两个子群的正确数/总数
                    _valid_nodes = [n for n, idx in zip(nodes, idxs) if idx is not None]
                    # 匹配列表 预测正确True，错误False
                    _correct_list = (pred == valid_socs).cpu().tolist()
                    for _node, _ok in zip(_valid_nodes, _correct_list):
                        # 判断节点的主父（由 PRIMARY_SOC_FG='Y' 确定）是否在被监督的训练集内
                        _flag = self._parent_supervised(_node, ntype)
                        if _flag is None:
                            continue
                        _key = "in" if _flag else "out"
                        # 节点的主父在被监督的训练集内
                        diag_total[_key] += 1
                        # 在被监督的训练集内，且预测正确
                        diag_correct[_key] += int(_ok)
                        if ntype in diag_total_lv:
                            diag_total_lv[ntype][_key] += 1
                            diag_correct_lv[ntype][_key] += int(_ok)
        
        val_acc = correct / max(1, total)
        val_loss = total_loss / max(1, num_batches)
        
        # 【新增】计算全局指标
        metrics = {
            "accuracy": val_acc,
            "macro_f1": f1_score(all_truths, all_preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(all_truths, all_preds, average="weighted", zero_division=0),
            "macro_precision": precision_score(all_truths, all_preds, average="macro", zero_division=0),
            "macro_recall": recall_score(all_truths, all_preds, average="macro", zero_division=0),
        }
        
        # 【新增】Per-level 指标
        for ntype in ["hlgt", "hlt", "pt", "llt"]:
            preds = per_level_preds.get(ntype, [])
            truths = per_level_truths.get(ntype, [])
            if len(preds) > 0:
                metrics[f"{ntype}_acc"] = accuracy_score(truths, preds)
                if len(set(truths)) > 1:
                    metrics[f"{ntype}_macro_f1"] = f1_score(
                        truths, preds, average="macro", zero_division=0
                    )
                else:
                    # 只有 1 类时 macro-F1 无意义，记 0（或记 acc 的等价）
                    metrics[f"{ntype}_macro_f1"] = 0.0
                metrics[f"{ntype}_n"] = len(preds)
            else:
                metrics[f"{ntype}_acc"] = 0.0
                metrics[f"{ntype}_macro_f1"] = 0.0
                metrics[f"{ntype}_n"] = 0                
        
        # 【诊断·方案①】结构线索通道：验证集按“主父是否在训练集内”分组的准确率对照。
        # 读法：两组接近 → 结构线索贡献有限；“主父∈train”明显更高 → 该差值可作为父链
        #       标签传播贡献的经验上界。适用层级为 HLT/PT/LLT（HLGT 主父为 SOC，不适用）。
        metrics["diag_parent_in_train_n"] = diag_total["in"]
        metrics["diag_parent_out_train_n"] = diag_total["out"]
        metrics["diag_parent_in_train_acc"] = (
            diag_correct["in"] / diag_total["in"] if diag_total["in"] > 0 else 0.0
        )
        metrics["diag_parent_out_train_acc"] = (
            diag_correct["out"] / diag_total["out"] if diag_total["out"] > 0 else 0.0
        )
        metrics["diag_parent_gap"] = (
            metrics["diag_parent_in_train_acc"] - metrics["diag_parent_out_train_acc"]
        )
        for _lv in self._DIAG_LEVELS:
            _acc_in = (diag_correct_lv[_lv]["in"] / diag_total_lv[_lv]["in"]
                       if diag_total_lv[_lv]["in"] > 0 else 0.0)
            _acc_out = (diag_correct_lv[_lv]["out"] / diag_total_lv[_lv]["out"]
                        if diag_total_lv[_lv]["out"] > 0 else 0.0)
            metrics[f"diag_{_lv}_in_train_n"] = diag_total_lv[_lv]["in"]
            metrics[f"diag_{_lv}_out_train_n"] = diag_total_lv[_lv]["out"]
            metrics[f"diag_{_lv}_in_train_acc"] = _acc_in
            metrics[f"diag_{_lv}_out_train_acc"] = _acc_out
            metrics[f"diag_{_lv}_gap"] = _acc_in - _acc_out

        # 【新增】保存混淆矩阵（可选，用于论文附图）
        metrics["confusion_matrix"] = confusion_matrix(all_truths, all_preds).tolist()
        
        # 计算层级相似度
        hierarchical_sim_dict = self.compute_hierarchical_intra_similarity(x_out)
        intra_sim_avg = hierarchical_sim_dict.get('average', 0.0)
        
        return val_loss, val_acc, intra_sim_avg, hierarchical_sim_dict, metrics

    # --------------------- 完整训练循环 ---------------------

    def train(self, verbose: bool = True, train_epoch_log: int = 1) -> Tuple[HeteroMedDRAEncoder, Dict]:
        """
        完整训练流程。
        
        参数:
            verbose (bool): 是否打印详细信息
            train_epoch_log: 每隔几轮打印一次训练信息
            
        返回:
            Tuple[HeteroMedDRAEncoder, Dict]: (训练好的模型, 训练历史)
        """
        cfg = self.training_config
        # 1. 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        # 2. 调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )

        if verbose:
            logger.info("=" * 60)
            logger.info("🚀 开始全量训练 (多层级对比 + SOC 分类)")
            logger.info(f"训练集: {len(self.train_indices):,} 个节点")
            logger.info(f"验证集: {len(self.val_indices):,} 个节点")
            logger.info(f"SOC 类别数: {self.num_socs}")
            logger.info(f"对比损失权重: {self.contrastive_config.contrastive_weight}")
            logger.info("=" * 60)

        # ✅ 模型选择：主判据为 self.es_primary_metric（默认 "acc" = 验证准确率），
        #    容差 self.es_tolerance 内优先选 gap_average 更大的模型；并启用塌缩护栏。
        self.best_val_acc = 0.0         # 历史最高准确率（仅记录/报告）
        self.best_primary = 0.0         # 主判据历史最优（用于“严格提升/容差”比较）
        self.best_gap = 0.0             #  gap_average of the selected model
        self.best_epoch = 0
        self.best_macro_f1 = 0.0        #  macro_f1 of the selected model
        patience_counter = 0            # 无严格提升的连续轮数
        self.best_val_loss = float("inf")   #  Best validation loss so far (used for the "still learning" decision)
        collapse_counter = 0            # 连续塌缩预警轮数（仅用于日志/报告）
        collapse_weight_total = 0       # 自上次“严格提升”以来累计计入 patience 的塌缩加权
        self._collapse_ref_intra = None # 塌缩判定参考 intra（随选中最佳权重更新）
        self._collapse_ref_gap = None   # 塌缩判定参考 gap（随选中最佳权重更新）
        start_time = time.time()

        # ---- 断点续训：检测断点并决定起始轮 ----
        _resume_on = bool(getattr(cfg, "resume", True)) and not os.environ.get("MEDDRA_NO_RESUME")
        _resume_every = max(0, int(getattr(cfg, "resume_every", 0) or 0))
        _resume_read, _resume_write = _resume_paths(cfg)   # 读取可来自配置；写入始终默认
        start_epoch = 0
        if not _resume_on:
            logger.info("断点续训: 已关闭（resume=False 或 环境变量 MEDDRA_NO_RESUME=1）")
        elif _resume_every <= 0:
            logger.info("断点续训: 未启用落盘（resume_every=0），本次不产生断点文件")
        elif os.path.exists(_resume_read):
            _res = self._load_resume(cfg, optimizer, scheduler)
            if _res is not None:
                start_epoch, patience_counter, collapse_counter, collapse_weight_total = _res
                logger.info(
                    f"♻️ 已从断点继续: 完成 {start_epoch} 轮，"
                    f"从 epoch {start_epoch + 1}/{cfg.epochs} 开始"
                    f"（当前最优 Acc={self.best_val_acc:.4f} @ Epoch {self.best_epoch + 1}）"
                )
            else:
                logger.info("断点续训: 断点未采用，从头训练")
        else:
            logger.info(
                f"断点续训: 开启（每 {_resume_every} 轮落盘到 {_resume_write}）；未发现断点，从头训练"
            )

        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()

            # 学习率预热
            if epoch < cfg.warmup_epochs:
                warmup_lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
            else:
                # # 4. 只在预热结束后步进调度器
                scheduler.step()

            # 3. 训练一个 epoch
            train_loss, train_acc, ce_loss, cl_loss = self.train_epoch(optimizer=optimizer, epoch=epoch)

            # 损失为空时停止训练
            if np.isnan(train_loss):
                logger.error(f"Epoch {epoch+1} 训练损失为 NaN，停止训练")
                break

            # 5. 验证
            val_loss, val_acc, intra_sim, hierarchical_sim_dict, metrics = self.evaluate()

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_ce_loss"].append(ce_loss)
            self.history["train_cl_loss"].append(cl_loss)
            self.history["epoch_times"].append(time.time() - epoch_start)
            self.history["learning_rates"].append(optimizer.param_groups[0]["lr"])
            self.history["intra_soc_sim"].append(intra_sim)  # 仅记录，不用于早停 各层级平均簇内相似度
            self.history["hierarchical_sim_dict"].append(hierarchical_sim_dict)  # 所有层级簇内相似度


            # 【新增】记录所有指标
            self.history["val_macro_f1"].append(metrics["macro_f1"])
            self.history["val_weighted_f1"].append(metrics["weighted_f1"])
            self.history["val_macro_precision"].append(metrics["macro_precision"])
            self.history["val_macro_recall"].append(metrics["macro_recall"])
            self.history["val_hlgt_acc"].append(metrics["hlgt_acc"])
            self.history["val_hlt_acc"].append(metrics["hlt_acc"])
            self.history["val_pt_acc"].append(metrics["pt_acc"])
            self.history["val_llt_acc"].append(metrics["llt_acc"])
            self.history["val_hlgt_macro_f1"].append(metrics["hlgt_macro_f1"])
            self.history["val_hlt_macro_f1"].append(metrics["hlt_macro_f1"])
            self.history["val_pt_macro_f1"].append(metrics["pt_macro_f1"])
            self.history["val_llt_macro_f1"].append(metrics["llt_macro_f1"])

            # ================= 模型选择（早停） =================
            # 主判据：self.es_primary_metric（默认 "acc" = 验证准确率，与论文主表口径一致）；
            # 次判据：主指标处于容差带内时，优先选择 gap_average 更大的模型。
            # macro-F1 不再作为选择判据，仅作报告指标并参与容差带内的择优。
            gap_avg = float(hierarchical_sim_dict.get("gap_average", 0.0)) if hierarchical_sim_dict else 0.0
            primary = val_acc if self.es_primary_metric == "acc" else metrics["macro_f1"]

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc          # 仅用于记录/报告历史最高准确率

            # 本轮塌缩判定需与“本轮之前的已选最佳”比较，故先快照旧参考值
            _ref_intra_old = self._collapse_ref_intra
            _ref_gap_old = self._collapse_ref_gap

            is_strict = primary > self.best_primary + self.es_min_delta
            in_tolerance = (self.best_primary - primary) <= self.es_tolerance
            better_gap = gap_avg > self.best_gap + self.es_gap_eps

            # 验证损失是否仍在下降 —— 用于判断“模型是否仍在学习”
            loss_improved = val_loss < self.best_val_loss - self.es_loss_min_delta
            if loss_improved:
                self.best_val_loss = val_loss

            # 早停保护期：前 es_min_epochs 轮为高学习率探索期，只记录、不累计 patience
            _counting = (epoch + 1) > self.es_min_epochs

            # 计数规则：
            #   · 主判据严格提升            → patience 归零（真实进展）
            #   · 主判据未提升 且 验证损失也不再下降 → 算一轮无进展（patience += 1）
            #   · 主判据未提升 但 验证损失仍在下降   → 视为“仍在学习”，不累加（避免欠拟合阶段误停）
            #   · 保护期内                  → 不累加
            #   （容差择优不计为提升，否则 gap 的持续微增会让训练无法按 patience 收敛停止）
            if is_strict:
                self.best_primary = primary
                patience_counter = 0
                collapse_weight_total = 0    # 严格提升则清空塌缩加权
            elif _counting and not loss_improved:
                patience_counter += 1

            if is_strict or (in_tolerance and better_gap):
                self.best_gap = gap_avg
                self.best_macro_f1 = metrics["macro_f1"]
                self.best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                # 保存最佳指标（去掉 confusion_matrix），并附带 gap 与选择依据
                self.best_metrics = {
                    k: v for k, v in metrics.items() if k != "confusion_matrix"
                }
                self.best_metrics["gap_average"] = gap_avg
                self.best_metrics["es_primary"] = self.es_primary_metric
                self.best_metrics["es_selected_by"] = "strict" if is_strict else "tolerance_gap"
                self.best_metrics["selected_epoch"] = int(epoch + 1)
                # 把层级几何指标一并写入 best_metrics，使同一 checkpoint 自洽、便于论文取数
                if hierarchical_sim_dict:
                    _hs = hierarchical_sim_dict
                    self.best_metrics["sim_intra_average"] = float(_hs.get("average", 0.0))
                    self.best_metrics["sim_inter_average"] = float(_hs.get("inter_average", 0.0))
                    self.best_metrics["sim_gap_average"] = float(_hs.get("gap_average", 0.0))
                    self.best_metrics["sim_cousin_average"] = float(_hs.get("cousin_average", 0.0))
                    for _lv in ("HLGT", "HLT", "PT", "LLT"):
                        self.best_metrics[f"sim_intra_{_lv.lower()}"] = float(_hs.get(_lv, 0.0))
                        self.best_metrics[f"sim_gap_{_lv.lower()}"] = float(_hs.get(f"gap_{_lv}", 0.0))
                # 选中新最佳权重时，同步刷新塌缩判定的参考几何
                self._collapse_ref_intra = intra_sim
                self._collapse_ref_gap = gap_avg
                if verbose:
                    logger.info(
                        f"  ✅ 新最佳[{self.best_metrics['es_selected_by']}] "
                        f"(Acc: {val_acc:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, "
                        f"Weighted-F1: {metrics['weighted_f1']:.4f}, Gap: {gap_avg:.4f})"
                    )

            # ================= 塌缩护栏（相对“已选最佳权重”的几何恶化） =================
            # 判据：intra 相对参考值上升，且 gap 相对参考值显著下降（降幅需超过 max(collapse_eps, 10%×参考gap)）。
            # 不使用任何绝对阈值——强语义特征下 intra 可长期处于 0.9+，绝对值高并不等于塌缩；
            # 只有“越挤越近(intra↑) 且 越分不开(gap↓)”的方向性恶化才算塌缩轮；intra↓ 且 gap↑（几何改善）不触发。
            # 计数方式（与早停共用同一条停止规则）：塌缩轮**等价计为 collapse_weight 轮无进展**——
            #   即本轮已有的 +1，再额外 +(collapse_weight − 1)，合计恰为 collapse_weight；不单独设停止阈值。
            # 等效反应速度 ≈ cfg.patience / collapse_weight 轮（15 / 3 = 5 轮）。
            if _ref_intra_old is not None and _ref_gap_old is not None:
                intra_worse = intra_sim > _ref_intra_old + self.collapse_eps
                gap_thr = max(self.collapse_eps, self.collapse_gap_ratio * abs(_ref_gap_old))
                gap_worse = gap_avg < _ref_gap_old - gap_thr
                if intra_worse and gap_worse:
                    collapse_counter += 1
                    _w = max(1, int(self.collapse_weight))
                    if _counting:
                        patience_counter += _w - 1
                        collapse_weight_total += _w - 1
                    if verbose:
                        logger.warning(
                            f"  ⚠️ 塌缩预警 {collapse_counter} 次"
                            f"（本轮按 ×{_w} 计入 patience {patience_counter}/{cfg.patience}"
                            f"{'' if _counting else '，保护期内不计数'}）"
                            f" (intra 参考 {_ref_intra_old:.4f}->{intra_sim:.4f}, "
                            f"gap 参考 {_ref_gap_old:.4f}->{gap_avg:.4f}, 需降幅>{gap_thr:.4f})"
                        )
                else:
                    collapse_counter = 0

            # 打印进度
            if verbose and ((epoch + 1) % train_epoch_log == 0 or epoch == 0 or epoch == cfg.epochs - 1):
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} "
                    f"({self.history['epoch_times'][-1]:.2f}s)"
                )
                logger.info(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                logger.info(
                    f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Macro-F1: {metrics['macro_f1']:.4f}, "
                    f"Weighted-F1: {metrics['weighted_f1']:.4f}"
                )
                logger.info(
                    f"  Per-level Acc: HLGT={metrics['hlgt_acc']:.4f}, "
                    f"HLT={metrics['hlt_acc']:.4f}, "
                    f"PT={metrics['pt_acc']:.4f}, "
                    f"LLT={metrics['llt_acc']:.4f}"
                )
                logger.info(
                    f"  Per-level Macro-F1: HLGT={metrics['hlgt_macro_f1']:.4f}, "
                    f"HLT={metrics['hlt_macro_f1']:.4f}, "
                    f"PT={metrics['pt_macro_f1']:.4f}, "
                    f"LLT={metrics['llt_macro_f1']:.4f}"
                )
                logger.info(
                    f"  【诊断】主父∈train: {metrics.get('diag_parent_in_train_acc', 0):.4f} "
                    f"(n={metrics.get('diag_parent_in_train_n', 0)}), "
                    f"主父∉train: {metrics.get('diag_parent_out_train_acc', 0):.4f} "
                    f"(n={metrics.get('diag_parent_out_train_n', 0)}), "
                    f"Gap={metrics.get('diag_parent_gap', 0):+.4f}"
                )
                logger.info(f"  Avg Hierarchical Sim: {intra_sim:.4f}")
                
                # 打印各层级相似度详情
                if hierarchical_sim_dict:
                    logger.info(
                        f"  Hierarchical Sim: HLGT={hierarchical_sim_dict.get('HLGT', 0):.4f}, "
                        f"HLT={hierarchical_sim_dict.get('HLT', 0):.4f}, "
                        f"PT={hierarchical_sim_dict.get('PT', 0):.4f}, "
                        f"LLT={hierarchical_sim_dict.get('LLT', 0):.4f}"
                    )
                    # 打印加权平均和均匀平均
                    logger.info(
                        f"  Weighted Avg: {hierarchical_sim_dict.get('average', 0):.4f}, "
                        f"Uniform Avg: {hierarchical_sim_dict.get('uniform_average', 0):.4f}"
                    )
                    # 【新增】采样估计的 inter（跨簇）与 gap（intra-inter）对照
                    logger.info(
                        f"  Inter(跨簇采样): HLGT={hierarchical_sim_dict.get('inter_HLGT', 0):.4f}, "
                        f"HLT={hierarchical_sim_dict.get('inter_HLT', 0):.4f}, "
                        f"PT={hierarchical_sim_dict.get('inter_PT', 0):.4f}, "
                        f"LLT={hierarchical_sim_dict.get('inter_LLT', 0):.4f}, "
                        f"Avg={hierarchical_sim_dict.get('inter_average', 0):.4f}"
                    )
                    logger.info(
                        f"  Gap(intra-inter): HLGT={hierarchical_sim_dict.get('gap_HLGT', 0):.4f}, "
                        f"HLT={hierarchical_sim_dict.get('gap_HLT', 0):.4f}, "
                        f"PT={hierarchical_sim_dict.get('gap_PT', 0):.4f}, "
                        f"LLT={hierarchical_sim_dict.get('gap_LLT', 0):.4f}, "
                        f"Avg={hierarchical_sim_dict.get('gap_average', 0):.4f}"
                    )
                    # 【新增】表兄弟硬负样本（同祖父、异主父）相似度对照
                    logger.info(
                        f"  Cousin(表兄弟硬负): HLGT={hierarchical_sim_dict.get('cousin_HLGT', 0):.4f}, "
                        f"HLT={hierarchical_sim_dict.get('cousin_HLT', 0):.4f}, "
                        f"PT={hierarchical_sim_dict.get('cousin_PT', 0):.4f}, "
                        f"LLT={hierarchical_sim_dict.get('cousin_LLT', 0):.4f}, "
                        f"Avg={hierarchical_sim_dict.get('cousin_average', 0):.4f}, "
                        f"GapCousin={hierarchical_sim_dict.get('gap_cousin_average', 0):.4f}"
                    )
                    # 打印有效节点数
                    eff_counts = hierarchical_sim_dict.get('effective_counts', {})
                    if eff_counts:
                        logger.info(
                            f"  Effective Counts: HLGT={eff_counts.get('HLGT', 0)}, "
                            f"HLT={eff_counts.get('HLT', 0)}, "
                            f"PT={eff_counts.get('PT', 0)}, "
                            f"LLT={eff_counts.get('LLT', 0)}"
                        )
                
                logger.info(
                    f"  CE: {ce_loss:.4f}, CL: {cl_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                logger.info(f"  Patience: {patience_counter}/{cfg.patience}")

            # ---- 断点续训：定期落盘（本轮状态已全部更新，续训将从下一轮开始）----
            if _resume_on and _resume_every > 0 and (epoch + 1) % _resume_every == 0:
                self._save_resume(cfg, optimizer, scheduler, epoch,
                                  patience_counter, collapse_counter, collapse_weight_total)

            # 统一停止规则：patience 计满即停（塌缩轮已按 collapse_weight 倍加速计入）
            if patience_counter >= cfg.patience:
                if verbose:
                    _reason = "塌缩加速" if collapse_weight_total > 0 else "常规无提升"
                    logger.info(
                        f"\n🛑 早停于 epoch {epoch+1}（{_reason}），"
                        f"patience={patience_counter}/{cfg.patience}"
                        f"（其中塌缩加权 {collapse_weight_total}）"
                        f"，best Val Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch+1})"
                    )
                break

        # 训练正常结束（含早停 / NaN 退出）→ 删除断点文件，避免下次运行被误当作"未完成的实验"
        if _resume_on and os.path.exists(_resume_write):
            try:
                os.remove(_resume_write)
                logger.info("🧹 训练已结束，断点文件已删除（无需续训）")
            except OSError as e:
                logger.warning(f"删除断点文件失败（可手动删除 {_resume_write}）: {e}")
        if _resume_read != RESUME_FILE:
            logger.info(f"ℹ️ 配置指定的续训文件保留未删除: {_resume_read}")

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(DEVICE)

        # ---- JBI 投稿：独立测试集评估（仅最终评估一次，不参与早停/模型选择）----
        self.test_metrics = {}
        if len(self.test_indices) > 0:
            _t_loss, _t_acc, _t_intra, _t_sim, _t_metrics = self.evaluate(self.test_indices)
            if _t_metrics:
                # 剥离混淆矩阵（过大，不写 JSON）；其余指标原样保留
                self.test_metrics = {k: v for k, v in _t_metrics.items() if k != "confusion_matrix"}
                self.test_metrics["test_accuracy"] = _t_acc
                self.test_metrics["test_loss"] = _t_loss
                self.test_metrics["selected_epoch"] = int(self.best_epoch + 1)
                self.test_metrics["num_test_nodes"] = int(len(self.test_indices))
                try:
                    import json as _tm_json
                    _tm_tag = str(globals().get("_MODEL_TAG", "model"))
                    with open(f"test_metrics_{_tm_tag}.json", "w", encoding="utf-8") as _tm_f:
                        _tm_json.dump(self.test_metrics, _tm_f, ensure_ascii=False, indent=2)
                    logger.info(f"\U0001f4c4 独立测试集指标已保存: test_metrics_{_tm_tag}.json")
                except Exception as _tm_e:
                    logger.warning(f"\u26a0\ufe0f 独立测试集指标写 JSON 失败（仍在 checkpoint 内）: {_tm_e}")
                if verbose:
                    logger.info(
                        f"\n🧪 独立测试集评估（{len(self.test_indices):,} 节点）："
                        f"Acc={_t_metrics.get('accuracy', 0):.4f}, "
                        f"Macro-F1={_t_metrics.get('macro_f1', 0):.4f}, "
                        f"Weighted-F1={_t_metrics.get('weighted_f1', 0):.4f}"
                    )

        total_time = time.time() - start_time
        if verbose:
            logger.info(f"\n✅ 训练完成! 总耗时: {total_time / 60:.2f} 分钟")
            logger.info(
                f"选中权重: Epoch {self.best_epoch + 1}"
                f"（Acc={self.best_metrics.get('accuracy', 0):.4f}）；"
                f"历史最高验证准确率: {self.best_val_acc:.4f}"
            )
            if self.best_metrics:
                logger.info(
                    f"选中模型[{self.best_metrics.get('es_selected_by', 'strict')}] "
                    f"(主判据={self.best_metrics.get('es_primary', 'acc')}): "
                    f"Acc={self.best_metrics.get('accuracy', 0):.4f}, "
                    f"Macro-F1={self.best_metrics.get('macro_f1', 0):.4f}, "
                    f"Gap={self.best_metrics.get('gap_average', 0):.4f}"
                )

        

        return self.model, self.history




    # --------------------- 生成嵌入 ---------------------

    def generate_embeddings(self) -> Dict[str, pd.DataFrame]:
        """
        为所有节点生成最终嵌入，并返回各层级 DataFrame。
        
        返回:
            Dict[str, pd.DataFrame]: 键为层级大写或 'all'，值为 DataFrame
        """
        # 评估模型
        self.model.eval()
        # 特征字典
        x_dict = {
            nt: self.data[nt].x for nt in self.node_types if hasattr(self.data[nt], "x")
        }
        # 边字典
        edge_dict = {
            et: self.data[et].edge_index
            for et in self.edge_types
            if hasattr(self.data[et], "edge_index")
        }

        with torch.no_grad():
            x_out, _, _ = self.model(x_dict, edge_dict)

        result = {}
        all_embs, all_codes, all_levels = [], [], []

        for nt in self.node_types:
            if nt in x_out and hasattr(self.data[nt], "node_codes"):
                codes = self.data[nt].node_codes
                embs = x_out[nt].cpu().numpy()
                df = pd.DataFrame(
                    embs,
                    index=codes,
                    columns=[f"dim_{i}" for i in range(self.model.embedding_dim)],
                )
                df["level"] = [self.processor.get_node_level(c) for c in codes]
                df["node_name"] = [self.processor.get_node_name(c) for c in codes]
                # 记录每层的 DataFrame
                result[nt.upper()] = df

                all_embs.append(embs)
                all_codes.extend(codes)
                # 记录每个节点的层级  - 每行
                all_levels.extend([self.processor.get_node_level(c) for c in codes])

        if all_embs:
            # 按行堆叠
            all_embs = np.vstack(all_embs)
            # 最终数据
            full_df = pd.DataFrame(
                all_embs,
                index=all_codes,
                columns=[f"dim_{i}" for i in range(self.model.embedding_dim)],
            )
            # 增加level列
            full_df["level"] = all_levels
            full_df["node_name"] = [self.processor.get_node_name(c) for c in all_codes]
            # 全部层级数据
            result["all"] = full_df

        logger.info(f"生成了 {len(result)} 个层级的嵌入")
        return result


# =============================================================================
# 7. 可视化工具
# =============================================================================

class Visualizer:
    """
    训练过程可视化和嵌入可视化工具。
    所有图表标题、轴标签、图例等均使用英文，确保在无中文字体环境下正常显示。
    """
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: str = "training_history.png") -> None:
        """
        绘制训练历史曲线。
        """
        if not history.get("train_loss"):
            logger.warning("无训练历史数据，无法绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("MedDRA Training History", fontsize=16, fontweight="bold")

        # ---- 1. 损失曲线 ---- 
        ax1 = axes[0, 0]
        ax1.plot(history["train_loss"], label="Train Loss", linewidth=2, color="blue")
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=2, color="red", linestyle="--")
        if history.get("train_ce_loss"):
            ax1.plot(history["train_ce_loss"], label="CE Loss", linewidth=1.5, color="cyan", linestyle=":")
        if history.get("train_cl_loss"):
            ax1.plot(history["train_cl_loss"], label="CL Loss", linewidth=1.5, color="orange", linestyle=":")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ---- 2. 准确率曲线 ----
        ax2 = axes[0, 1]
        ax2.plot(history["train_acc"], label="Train Acc", linewidth=2, color="blue")
        ax2.plot(history["val_acc"], label="Val Acc", linewidth=2, color="red", linestyle="--")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ---- 3. 簇内相似度 ----
        ax3 = axes[1, 0]
        if history.get("intra_soc_sim"):
            ax3.plot(history["intra_soc_sim"], label="Average-Cluster Similarity", linewidth=2, color="green")
            if history["hierarchical_sim_dict"]: 
                 # 提取各层级数据
                ax3.plot([d.get('HLGT', 0) for d in history["hierarchical_sim_dict"]], label="HLGT-Cluster Similarity", linewidth=2, color="red")
                ax3.plot([d.get('HLT', 0) for d in history["hierarchical_sim_dict"]], label="HLT-Cluster Similarity", linewidth=2, color="blue")
                ax3.plot([d.get('PT', 0) for d in history["hierarchical_sim_dict"]], label="PT-Cluster Similarity", linewidth=2, color="cyan")
                ax3.plot([d.get('LLT', 0) for d in history["hierarchical_sim_dict"]], label="LLT-Cluster Similarity", linewidth=2, color="orange")

            best_idx = np.argmax(history["intra_soc_sim"])
            best_val = history["intra_soc_sim"][best_idx]
            ax3.scatter(best_idx, best_val, color="green", s=100, zorder=5)
            # 最佳节点标注
            ax3.annotate(
                f"Best: {best_val:.4f}",
                xy=(best_idx, best_val),
                xytext=(best_idx + 1, best_val - 0.05),
                arrowprops=dict(arrowstyle="->", color="green"),
            )
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Cosine Similarity")
            ax3.set_title("Cluster Similarity")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # ---- 4. 学习率 ----
        ax4 = axes[1, 1]
        if history.get("learning_rates"):
            ax4.plot(history["learning_rates"], linewidth=2, color="purple")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Learning Rate")
            ax4.set_title("Learning Rate Schedule")
            ax4.set_yscale("log")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"训练历史图已保存: {save_path}")
        plt.show()

    @staticmethod
    def visualize_embeddings_by_all_soc(
        level_embeddings: Dict[str, pd.DataFrame],
        processor: HeteroMedDRADataProcessor,  # 需要传入 processor 获取 SOC 映射
        sample_size: int = 3000,
        method: str = "tsne",
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        按 SOC 着色可视化嵌入，评估分类效果。
        """
        if not level_embeddings:
            return None
        
        logger.info(f"使用 {method.upper()} 按 SOC 可视化嵌入...")
        all_embs, all_soc_labels = [], []
        
        # 获取所有非 all 的层级
        level_names = [l for l in level_embeddings.keys() if l != "all"]
        
        for level in level_names:
            df = level_embeddings[level]
            dim_cols = [c for c in df.columns if c.startswith("dim_")]
            if not dim_cols or len(df) == 0:
                continue
            
            embs = df[dim_cols].values
            # 获取每个节点对应的 SOC 代码
            soc_codes = []
            for node_code in df.index:
                soc = processor.get_soc_node_code(node_code)
                soc_codes.append(soc if soc is not None else -1)
            
            # 采样
            n_samples = min(sample_size // len(level_names), len(embs))
            if len(embs) > n_samples and n_samples > 0:
                idx = np.random.choice(len(embs), n_samples, replace=False)
                embs = embs[idx]
                soc_codes = [soc_codes[i] for i in idx]
            
            all_embs.append(embs)
            all_soc_labels.extend(soc_codes)
        
        if not all_embs:
            return None
        
        all_embs = np.vstack(all_embs)
        all_soc_labels = np.array(all_soc_labels)
        
        # 降维
        if method.lower() == "tsne":
            perplexity = min(30, max(5, len(all_embs) - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        emb_2d = reducer.fit_transform(all_embs)
        
        # 可视化
        plt.figure(figsize=(14, 10))
        unique_socs = np.unique(all_soc_labels)
        # 过滤掉 -1（无效标签）
        unique_socs = unique_socs[unique_socs >= 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_socs)))
        
        for i, soc in enumerate(unique_socs):
            mask = all_soc_labels == soc
            if mask.sum() > 0:
                soc_name = processor.get_node_name(soc)
                plt.scatter(
                    emb_2d[mask, 0],
                    emb_2d[mask, 1],
                    c=[colors[i]],
                    label=soc_name,
                    alpha=0.6,
                    s=15,
                )
        
        plt.title(f"MedDRA Embedding Visualization by SOC ({method.upper()})", 
                fontsize=16, fontweight="bold")
        plt.xlabel(f"{method.upper()} Dimension 1")
        plt.ylabel(f"{method.upper()} Dimension 2")
        plt.legend(loc="best", fontsize=8, ncol=3)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        
        return emb_2d

    @staticmethod
    def visualize_embeddings_by_part_soc(
        level_embeddings: Dict[str, pd.DataFrame],
        processor: HeteroMedDRADataProcessor,
        sample_size: int = 3000,
        method: str = "tsne",
        save_path: Optional[str] = None,
        min_samples_per_soc: int = 5,  # 每个SOC最少采样数
        max_socs_to_show: int = 20,    # 最多显示多少个SOC
    ) -> Optional[np.ndarray]:
        """
        按 SOC 着色可视化嵌入，评估分类效果。
        
        优化点：
        1. 按 SOC 分层采样，确保每个 SOC 都有代表性样本
        2. 自动过滤样本数过少的 SOC
        3. 使用更好的颜色映射
        4. 添加统计信息
        """
        if not level_embeddings:
            logger.warning("无嵌入数据可供可视化")
            return None
        
        logger.info(f"使用 {method.upper()} 按 SOC 可视化嵌入...")
        
        # 1. 收集所有数据并按 SOC 分组
        soc_to_samples = defaultdict(list)
        level_names = [l for l in level_embeddings.keys() if l != "all"]
        
        for level in level_names:
            df = level_embeddings[level]
            dim_cols = [c for c in df.columns if c.startswith("dim_")]
            if not dim_cols or len(df) == 0:
                continue
            
            embs = df[dim_cols].values
            # 获取每个节点对应的 SOC 代码
            for i, node_code in enumerate(df.index):
                soc = processor.get_soc_node_code(node_code)
                if soc is not None and soc >= 0:
                    soc_to_samples[soc].append(embs[i])
        
        if not soc_to_samples:
            logger.warning("未找到有效的 SOC 标签")
            return None
        
        # 2. 统计信息
        soc_counts = {soc: len(samples) for soc, samples in soc_to_samples.items()}
        logger.info(f"共有 {len(soc_counts)} 个 SOC，总样本数: {sum(soc_counts.values())}")
        
        # 3. 过滤样本数过少的 SOC
        valid_socs = [soc for soc, count in soc_counts.items() if count >= min_samples_per_soc]
        if len(valid_socs) < len(soc_counts):
            logger.info(f"过滤掉 {len(soc_counts) - len(valid_socs)} 个样本数 < {min_samples_per_soc} 的 SOC")
        
        # 4. 如果 SOC 太多，只显示主要的
        if len(valid_socs) > max_socs_to_show:
            # 按样本数排序，选择样本最多的 SOC
            valid_socs.sort(key=lambda x: soc_counts[x], reverse=True)
            valid_socs = valid_socs[:max_socs_to_show]
            logger.info(f"只显示样本最多的 {max_socs_to_show} 个 SOC")
        
        # 5. 按 SOC 分层采样
        all_embs = []
        all_soc_labels = []
        
        # 计算每个 SOC 的采样配额（按比例分配）
        total_samples = sum(soc_counts[soc] for soc in valid_socs)
        
        for soc in valid_socs:
            samples = np.array(soc_to_samples[soc])
            n_available = len(samples)
            
            # 计算该 SOC 的采样数（按比例，但保证最小值）
            n_samples = max(
                min_samples_per_soc,  # 至少 min_samples_per_soc 个
                int(sample_size * (n_available / total_samples))  # 按比例分配
            )
            n_samples = min(n_samples, n_available)  # 不能超过可用样本数
            
            # 采样
            if n_available > n_samples and n_samples > 0:
                idx = np.random.choice(n_available, n_samples, replace=False)
                sampled_embs = samples[idx]
            else:
                sampled_embs = samples
            
            all_embs.append(sampled_embs)
            all_soc_labels.extend([soc] * len(sampled_embs))
            
            logger.debug(f"SOC {soc}: {n_available} → {len(sampled_embs)} 个样本")
        
        if not all_embs:
            logger.warning("无有效数据")
            return None
        
        # 6. 合并数据
        all_embs = np.vstack(all_embs)
        all_soc_labels = np.array(all_soc_labels)
        logger.info(f"可视化节点总数: {len(all_embs)}")
        
        # 7. 降维
        if method.lower() == "tsne":
            # 动态调整 perplexity
            n_samples = len(all_embs)
            perplexity = min(50, max(5, n_samples // 10))
            logger.info(f"t-SNE perplexity: {perplexity}")
            reducer = _make_tsne(perplexity=perplexity)   # 兼容 sklearn>=1.5（n_iter→max_iter）
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        logger.info("执行降维...")
        emb_2d = reducer.fit_transform(all_embs)
        
        # 8. 可视化（优化版）
        unique_socs = np.unique(all_soc_labels)
        n_socs = len(unique_socs)
        
        # 使用更好的颜色映射
        if n_socs <= 10:
            # 少量 SOC 使用鲜艳的颜色
            colors = plt.cm.tab10(np.linspace(0, 1, n_socs))
        elif n_socs <= 20:
            # 中等数量使用 tab20
            colors = plt.cm.tab20(np.linspace(0, 1, n_socs))
        else:
            # 大量 SOC 使用渐变色
            colors = plt.cm.Set3(np.linspace(0, 1, n_socs))
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 主图：所有 SOC
        for i, soc in enumerate(unique_socs):
            mask = all_soc_labels == soc
            if mask.sum() > 0:
                soc_name = processor.get_node_name(soc)
                # 截断过长的名称
                if len(soc_name) > 20:
                    soc_name = soc_name[:17] + "..."
                
                ax1.scatter(
                    emb_2d[mask, 0],
                    emb_2d[mask, 1],
                    c=[colors[i]],
                    label=f"{soc_name} ({mask.sum()})",
                    alpha=0.7,
                    s=20,
                    edgecolors='white',
                    linewidth=0.5,
                )
        
        ax1.set_title(f"MedDRA Embedding by SOC ({method.upper()})\n{n_socs} SOCs, {len(all_embs)} samples", 
                    fontsize=14, fontweight="bold")
        ax1.set_xlabel(f"{method.upper()} Dimension 1")
        ax1.set_ylabel(f"{method.upper()} Dimension 2")
        ax1.legend(loc="best", fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # 子图：显示聚类紧凑度（可选）
        # 计算每个 SOC 的类内距离
        intra_distances = []
        soc_names_short = []
        for soc in unique_socs:
            mask = all_soc_labels == soc
            if mask.sum() > 1:
                soc_embs = emb_2d[mask]
                # 计算到中心的平均距离
                center = np.mean(soc_embs, axis=0)
                distances = np.linalg.norm(soc_embs - center, axis=1)
                intra_distances.append(np.mean(distances))
                soc_names_short.append(processor.get_node_name(soc)[:15])
        
        if intra_distances:
            # 按距离排序
            sorted_indices = np.argsort(intra_distances)[::-1]  # 从大到小
            sorted_names = [soc_names_short[i] for i in sorted_indices]
            sorted_distances = [intra_distances[i] for i in sorted_indices]
            
            # 只显示前20个
            if len(sorted_distances) > 20:
                sorted_names = sorted_names[:20]
                sorted_distances = sorted_distances[:20]
            
            ax2.barh(sorted_names, sorted_distances, color='steelblue', alpha=0.7)
            ax2.set_xlabel("Average Intra-SOC Distance (Lower = Better)")
            ax2.set_title("SOC Clustering Compactness\n(Lower = More Compact)", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"可视化图已保存: {save_path}")
        plt.show()
        
        # 9. 输出统计信息
        logger.info("\n" + "="*50)
        logger.info("SOC 聚类统计:")
        logger.info("="*50)
        for soc in unique_socs:
            mask = all_soc_labels == soc
            n_samples = mask.sum()
            if n_samples > 1:
                soc_embs = emb_2d[mask]
                center = np.mean(soc_embs, axis=0)
                max_dist = np.max(np.linalg.norm(soc_embs - center, axis=1))
                logger.info(f"  {processor.get_node_name(soc)}: {n_samples} 样本, 最大半径: {max_dist:.3f}")
        
        return emb_2d


    @staticmethod
    def visualize_soc_confusion(
        level_embeddings: Dict[str, pd.DataFrame],
        processor: HeteroMedDRADataProcessor,
        save_path: str = "visualize_soc_confusion.png"
    ) -> None:
        """
        可视化 SOC 之间的混淆程度（相似度矩阵）
        """
        # 1. 收集每个 SOC 的中心嵌入
        soc_centers = {}
        level_names = [l for l in level_embeddings.keys() if l != "all"]
        
        for level in level_names:
            df = level_embeddings[level]
            dim_cols = [c for c in df.columns if c.startswith("dim_")]
            if not dim_cols:
                continue
            
            embs = df[dim_cols].values
            for i, node_code in enumerate(df.index):
                soc = processor.get_soc_node_code(node_code)
                if soc is not None and soc >= 0:
                    if soc not in soc_centers:
                        soc_centers[soc] = []
                    soc_centers[soc].append(embs[i])
        
        # 2. 计算每个 SOC 的中心
        soc_list = []
        center_embs = []
        for soc, samples in soc_centers.items():
            if len(samples) >= 5:  # 至少5个样本才计算中心
                soc_list.append(soc)
                center_embs.append(np.mean(samples, axis=0))
        
        if len(soc_list) < 2:
            logger.warning("SOC 数量不足，无法绘制混淆矩阵")
            return
        
        center_embs = np.array(center_embs)
        
        # 3. 计算 SOC 间相似度
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(center_embs)
        
        # 4. 可视化
        plt.figure(figsize=(12, 10))
        
        # 获取 SOC 名称
        soc_names = [processor.get_node_name(soc) for soc in soc_list]
        # 截断名称
        soc_names_short = [name[:20] + "..." if len(name) > 20 else name for name in soc_names]
        
        # 绘制热力图
        sns.heatmap(
            sim_matrix,
            xticklabels=soc_names_short,
            yticklabels=soc_names_short,
            annot=False,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Cosine Similarity'},
            square=True,
        )
        
        plt.title("SOC Similarity Matrix\n(Lower similarity = Better separation)", 
                fontsize=16, fontweight="bold")
        plt.xlabel("SOC")
        plt.ylabel("SOC")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"可视化 SOC 之间的混淆程度（相似度矩阵）已保存: {save_path}")
        plt.show()

# =============================================================================
# 8. 模型导出工具
# =============================================================================

class ModelExporter:
    """模型导出工具。"""
    
    @staticmethod
    def save_embeddings(level_embeddings: Dict[str, pd.DataFrame], output_prefix: str = "meddra_embeddings") -> None:
        """将各层级嵌入保存为 CSV 文件。"""
        if not level_embeddings:
            logger.warning("嵌入数据为空，无法导出")
            return

        for level, df in level_embeddings.items():
            if level == "all":
                continue
            df_with_node = df.reset_index().rename(columns={"index": "nodeCode"})
            filename = f"{output_prefix}_{level}.csv"
            df_with_node.to_csv(filename, index=False, encoding="utf-8-sig")
            logger.info(f"✅ {level} 嵌入已保存: {filename}")

        if "all" in level_embeddings:
            all_df = level_embeddings["all"].reset_index().rename(columns={"index": "nodeCode"})
            all_df.to_csv(f"{output_prefix}_all.csv", index=False, encoding="utf-8-sig")
            logger.info(f"✅ 全部嵌入已保存: {output_prefix}_all.csv")

    @staticmethod
    def save_model(model: HeteroMedDRAEncoder, filepath: str = "best_meddra_encoder.pth") -> None:
        """保存模型权重。"""
        torch.save(model.state_dict(), filepath)
        logger.info(f"✅ 模型权重已保存: {filepath}")

    @staticmethod
    # 【预留·未接线】权重加载接口：当前三个训练流程均未调用（仅为后续推理/续训预留）。
    def load_model(model: HeteroMedDRAEncoder, filepath: str = "best_meddra_encoder.pth") -> HeteroMedDRAEncoder:
        """加载模型权重。"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        state_dict = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info(f"✅ 模型权重已加载: {filepath}")
        return model.to(DEVICE)


    @staticmethod
    def save_checkpoint(
        model: HeteroMedDRAEncoder,
        processor: HeteroMedDRADataProcessor,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        contrastive_config: ContrastiveConfig,
        best_metrics: Dict,
        filepath: str = "best_checkpoint.pt",
        test_metrics: Optional[Dict] = None,
    ) -> None:
        """保存完整 checkpoint，包含模型、配置、元数据（含可选的独立测试集指标）。"""
        from dataclasses import asdict

        checkpoint = {
            # 模型权重
            "model_state_dict": model.state_dict(),
            # 配置
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "contrastive_config": asdict(contrastive_config),
            # 模型结构信息
            "node_types": list(model.node_types),
            "edge_types": list(model.edge_types),
            "num_socs": model.num_socs,
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            # 数据处理信息
            "level_map": processor.level_map,
            "level_names": processor.level_names,
            "level_nodes": {k: list(v) for k, v in processor.level_nodes.items()},
            "soc_list": processor.level_nodes.get("SOC", []),
            "num_all_nodes": processor.num_all_nodes,
            # 最佳指标
            "best_metrics": best_metrics,
            # 版本
            "cache_version": GraphCacheManager.CACHE_VERSION,
            "build_hierarchy_version": BUILD_HIERARCHY_VERSION,   # 【版本差异检测】
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 【JBI 投稿】独立测试集指标（可选，三分切分时写入）
        if test_metrics is not None:
            checkpoint["test_metrics"] = test_metrics
        torch.save(checkpoint, filepath)
        logger.info(f"✅ 完整 checkpoint 已保存: {filepath}")

    @staticmethod
    # 【预留·未接线】**完整 checkpoint 加载**接口（供推理 / 增量训练用），训练流程未调用。
    # 注意它与「自动断点续训」是两件事 —— 后者由 RESUME_FILE / _save_resume / _load_resume 实现：
    #   默认每 10 轮自动落盘、启动时自动续训、训练正常结束自动删除（见 TrainingConfig.resume）。
    # 另：其内部已加入「版本差异检测」，会校验 checkpoint 的 build_hierarchy_version。
    def load_checkpoint(filepath: str, device=DEVICE):
        """加载完整 checkpoint，重建模型。"""
        from dataclasses import fields

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint 不存在: {filepath}")

        ckpt = torch.load(filepath, map_location=device, weights_only=False)

        # 【版本差异检测】checkpoint 记录的是保存时的层级/标签构建版本；
        # 与当前代码不一致时给出明确告警（不直接报错），提示标签口径可能不同。
        _ckpt_ver = ckpt.get("build_hierarchy_version", None)
        if _ckpt_ver != BUILD_HIERARCHY_VERSION:
            logger.warning(
                f"⚠️ Checkpoint 版本不匹配：checkpoint={_ckpt_ver}，当前代码={BUILD_HIERARCHY_VERSION}"
                f" → 其标签/主父口径可能与当前代码不同，使用前请确认"
            )

        # 重建 ModelConfig（过滤掉未知字段）
        valid_fields = {f.name for f in fields(ModelConfig)}
        model_config = ModelConfig(**{
            k: v for k, v in ckpt["model_config"].items()
            if k in valid_fields
        })

        # 重建模型
        model = HeteroMedDRAEncoder(
            node_types=ckpt["node_types"],
            edge_types=ckpt["edge_types"],
            num_socs=ckpt["num_socs"],
            model_config=model_config,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(f"✅ Checkpoint 已加载: {filepath}")
        logger.info(f"   最佳指标: {ckpt.get('best_metrics', {})}")
        return model, ckpt



# =============================================================================
# 9. 记忆库管理（改进版：支持 Herding 采样）
# =============================================================================

def build_memory_bank(
    trainer: HeteroMedDRATrainer,
    k_per_soc: int = 10,
    output_file: str = "memory_bank.pkl",
    sampling_strategy: str = "herding",  # [改进] 'random' 或 'herding'
    seed: int = 42,
) -> List[Tuple[int, str, torch.Tensor, Optional[torch.Tensor]]]:
    """
    构建记忆库，每个 SOC 每个层级保留 k 个代表性样本。
    新增返回 soft labels（概率分布）用于蒸馏。
    采样策略：
        - 'random': 随机采样
        - 'herding': 选择离类中心最近的样本（更均匀覆盖）
    """
    logger.info(f"开始构建记忆库 (策略: {sampling_strategy})...")
    model = trainer.model
    model.eval()
    processor = trainer.processor
    data = trainer.data
    rng = np.random.RandomState(seed)

    x_dict = {
        nt: data[nt].x.to(DEVICE)
        for nt in trainer.node_types
        if hasattr(data[nt], "x")
    }
    edge_dict = {
        et: data[et].edge_index.to(DEVICE)
        for et in trainer.edge_types
        if hasattr(data[et], "edge_index")
    }

    with torch.no_grad():
        x_out, soc_logits, _ = model(x_dict, edge_dict)

    memory_bank = []
    soc_list = processor.level_nodes.get("SOC", [])

    for soc in tqdm(soc_list, desc="构建记忆库"):
        for level in ["HLGT", "HLT", "PT", "LLT"]:
            nodes = processor.get_nodes_by_level(level)
            soc_nodes = [n for n in nodes if processor.get_soc_node_code(n) == soc]
            if not soc_nodes:
                continue

            ntype = level.lower()
            # 获取嵌入
            node_embs = []
            valid_nodes = []
            for node in soc_nodes:
                idx = trainer._get_node_index(node, ntype)
                if idx is not None and ntype in x_out:
                    node_embs.append(x_out[ntype][idx].cpu())
                    valid_nodes.append(node)

            if not node_embs:
                continue

            node_embs = torch.stack(node_embs)  # (m, dim)
            n_available = len(valid_nodes)

            if n_available <= k_per_soc:
                sampled_nodes = valid_nodes
                sampled_embs = node_embs
            else:
                if sampling_strategy == 'herding':
                    # Herding: 选择离类中心最近的样本
                    center = node_embs.mean(dim=0)
                    # 计算每个样本到中心的欧氏距离（或余弦距离，这里用欧氏）
                    dist = torch.norm(node_embs - center, dim=1)
                    # 选择距离最小的 k 个
                    _, indices = torch.topk(dist, k_per_soc, largest=False)
                    indices = indices.tolist()
                    sampled_nodes = [valid_nodes[i] for i in indices]
                    sampled_embs = node_embs[indices]
                else:  # random
                    idx_sel = rng.choice(n_available, k_per_soc, replace=False)
                    # idx_sel = np.random.choice(n_available, k_per_soc, replace=False)
                    sampled_nodes = [valid_nodes[i] for i in idx_sel]
                    sampled_embs = node_embs[idx_sel]

            # 获取这些节点的 soft labels（用于蒸馏）
            # 目标：不强制旧节点的嵌入值不变，而是强制旧节点的输出概率分布不变。
            soft_labels = []
            for node in sampled_nodes:
                idx = trainer._get_node_index(node, ntype)
                if idx is not None and ntype in soc_logits:
                    # 1. 获取模型对当前节点的原始输出分数（logits）
                    logits = soc_logits[ntype][idx]
                    # 2. 通过 Softmax 函数将 logits 转换为概率分布
                    soft = F.softmax(logits, dim=-1).cpu()
                    soft_labels.append(soft)
                else:
                    soft_labels.append(None)

            # 存入记忆库：节点代码，节点类型，嵌入，soft (是一个概率分布向量, 一个一维向量, 长度等于 SOC(系统器官分类)的类别总数)
            for node, emb, soft in zip(sampled_nodes, sampled_embs, soft_labels):
                memory_bank.append((node, ntype, emb, soft))

    # 保存到文件
    with open(output_file, "wb") as f:
        pickle.dump(memory_bank, f)

    logger.info(f"✅ 记忆库构建完成，共 {len(memory_bank)} 个样本，保存至 {output_file}")
    return memory_bank




# =============================================================================
# 10. 特征缓存工具函数
# =============================================================================

# HF 权重文件判定标记（用于判断本地目录中是否已有可用模型）
_HF_WEIGHT_FILES = ("config.json", "pytorch_model.bin", "model.safetensors")


def load_text_encoder(model_name: str, local_dir: str = "", verbose: bool = True):
    """
    统一加载文本编码器（tokenizer + 模型），实现"下载一次、之后读本地"。

    行为：
        · local_dir 非空且目录中已有权重 → 离线从本地加载（local_files_only=True，零联网）；
        · local_dir 非空但目录中无权重   → 从 HF Hub 下载一次并 save_pretrained 到该目录，
                                          供后续运行离线复用；
        · local_dir 为空                → 保持原行为（按 HF Hub 名称加载，依赖 HF 自身缓存）。

    参数:
        model_name (str): HF Hub 模型名（如 "microsoft/BiomedNLP-PubMedBERT-..."）
        local_dir (str): 本地模型目录；为空则不做本地化
        verbose (bool): 是否打印加载来源

    返回:
        (tokenizer, text_encoder)：调用方负责 .to(DEVICE) / .eval() / 冻结参数
    """
    _local_ready = bool(local_dir) and os.path.isdir(local_dir) and any(
        os.path.exists(os.path.join(local_dir, _f)) for _f in _HF_WEIGHT_FILES
    )

    if _local_ready:
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        text_encoder = AutoModel.from_pretrained(local_dir, local_files_only=True)
        if verbose:
            logger.info(f"✅ 文本编码器：从本地目录离线加载（不联网）: {local_dir}")
        return tokenizer, text_encoder

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder = AutoModel.from_pretrained(model_name)
    if local_dir:
        try:
            os.makedirs(local_dir, exist_ok=True)
            tokenizer.save_pretrained(local_dir)
            text_encoder.save_pretrained(local_dir)
            if verbose:
                logger.info(f"📦 文本编码器已下载并保存到本地: {local_dir}（下次运行将离线读取）")
        except Exception as e:
            # 只读目录（如未先拷贝的 Kaggle Dataset）会走到这里；不影响本次运行
            logger.warning(f"保存文本编码器到 {local_dir} 失败（不影响本次运行）: {e}")
    elif verbose:
        logger.info(f"ℹ️ 未配置 text_model_dir，按 HF 名称加载（依赖 HuggingFace 缓存）: {model_name}")
    return tokenizer, text_encoder


def get_pubmedbert_features(
    processor: HeteroMedDRADataProcessor,
    cache_manager: GraphCacheManager,
    embedding_dim: int = 256,
    feature_type: str = "pubmedbert",  # 'pubmedbert' 或 'random'
    force_rebuild: bool = False,
    verbose: bool = True,
    model_local_dir: str = "",
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    获取或生成节点特征，支持 PubMedBERT 或随机特征（用于消融实验）。
    
    参数:
        processor (HeteroMedDRADataProcessor): 数据处理器
        cache_manager (GraphCacheManager): 缓存管理器
        embedding_dim (int): 嵌入维度（仅用于缓存键）
        feature_type (str): 特征类型，'pubmedbert' 或 'random'
        force_rebuild (bool): 是否强制重新生成
        verbose (bool): 是否打印详细信息
        model_local_dir (str): 文本编码器本地目录（空则按 HF Hub 名称加载）
        
    返回:
        Tuple[Dict[str, torch.Tensor], int]: 
            - features_dict: {node_type: (num_nodes, feature_dim)}
            - feature_dim: 特征维度（768 或 128）
    """
    # ---- 1. 随机特征模式（消融实验基线） ----
    if feature_type == "random":
        if verbose:
            logger.info("📝 生成随机特征（消融实验基线）...")
        
        features_dict = {}
        feature_dim = 128  # ✅ 随机特征维度
        
        for ntype in processor.level_names_lower:
            nodes = processor.level_nodes.get(ntype.upper(), [])
            if not nodes:
                continue
            # 生成随机特征（使用与之前相同的 One-hot + 噪声方式）- Level
            features = []
            level_idx = processor.level_map.get(ntype.upper(), 0)
            one_hot = torch.zeros(len(processor.level_map))
            one_hot[level_idx] = 1
            
            for _ in nodes:
                # 扩展 one-hot 到 feature_dim
                feature = one_hot.repeat(feature_dim // len(processor.level_map) + 1)[:feature_dim]
                # 使用较大的随机噪声（0.1），破坏对称性，防止所有点塌陷到一个地方
                feature = feature + 0.1 * torch.randn(feature_dim)
                feature = feature / (feature.norm() + 1e-8)
                features.append(feature)
            
            features_dict[ntype] = torch.stack(features)
            if verbose:
                logger.info(f"  {ntype}: {features_dict[ntype].shape} (随机特征)")
        
        return features_dict, feature_dim


    # 还可以增加其他的特征处理方式  比如完全随机、根据soc One-hot + 噪声方式
    
    # ---- 2. PubMedBERT 特征模式（原有逻辑） ----
    if not force_rebuild and cache_manager.has_pubmedbert_cache(processor.meddra_csv_path, embedding_dim):
        if verbose:
            logger.info("发现 PubMedBERT 特征缓存，正在加载...")
        cached_features = cache_manager.load_pubmedbert_features(processor.meddra_csv_path, embedding_dim)
        if cached_features is not None:
            all_match = True
            for ntype in processor.level_names_lower:
                if ntype in cached_features and hasattr(processor.graph_data[ntype], 'x'):
                    # 对比图节点数与缓存的特征节点数
                    if cached_features[ntype].shape[0] != processor.graph_data[ntype].x.shape[0]:
                        all_match = False
                        logger.warning(f"{ntype} 缓存节点数 ({cached_features[ntype].shape[0]}) "
                                       f"与图数据 ({processor.graph_data[ntype].x.shape[0]}) 不匹配，将重新生成")
                        break
            if all_match:
                return cached_features, 768
    
    if verbose:
        logger.info("📝 生成 PubMedBERT 语义特征...")

    # 获取分词器和编码器
    try:
        # 【本地优先】model_local_dir 非空时离线读本地；目录无权重则下载一次并落盘
        tokenizer, text_encoder = load_text_encoder(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            local_dir=model_local_dir,
            verbose=verbose,
        )
        text_encoder.to(DEVICE)
        text_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False
    except Exception as e:
        logger.error(f"加载 PubMedBERT 失败: {e}")
        return {}, 0

    features_dict = {}
    name_dict = processor.get_node_names_dict()

    for ntype in processor.level_names_lower:
        if ntype in name_dict and name_dict[ntype]:
            names = name_dict[ntype]
            all_embs = []
            for i in range(0, len(names), 256):
                batch_names = names[i:i + 256]
                inputs = tokenizer(
                    batch_names,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(DEVICE)
                with torch.no_grad():
                    # 编码
                    outputs = text_encoder(**inputs)
                    # 取值
                    cls_emb = outputs.last_hidden_state[:, 0, :]
                all_embs.append(cls_emb.cpu())

            # 当前层所有的编码结果
            node_embs = torch.cat(all_embs, dim=0)
            features_dict[ntype] = node_embs
            if verbose:
                logger.info(f"  {ntype}: {node_embs.shape}")

    # 释放文本编码器内存
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # 保存到缓存
    cache_manager.save_pubmedbert_features(processor.meddra_csv_path, features_dict, embedding_dim)

    return features_dict, 768




# =============================================================================
# 11. 训练入口函数
# =============================================================================

def log_run_banner(
    model_kind: str,
    aggregation: str,
    model_config,
    training_config,
    data_config=None,
    contrastive_config=None,
    graph_data=None,
    processor=None,
    feature_dim=None,
    verbose: bool = True,
) -> None:
    """
    打印“运行自证横幅”（A2）。

    目的：训练历史、权重与日志文件经常被改名、混用或误命名，仅凭文件名无法判断某次
    运行究竟是哪个模型（本项目已因此连续出错两次）。本横幅把“模型类型 + 聚合算子 +
    网络结构 + 文本特征 + 损失配置 + 优化配置 + 图规模”一次性以固定格式打印，使任何
    日志都能自证归属——检索关键字为 "RUN-BANNER"。

    实现说明：所有字段均以 getattr 读取，缺失时显示 "—"，从而不会因三个文件的配置类
    差异（如 noGNN 无 ContrastiveConfig、SAGE/GAT 无 feature_model）而中断训练。
    """
    if not verbose:
        return

    def _g(obj, name, default="—"):
        return getattr(obj, name, default) if obj is not None else default

    lines = [
        "🧬 RUN-BANNER 运行自证（判断归属于哪个模型请以此为准，勿依赖文件名）",
        f"   模型类型   : {model_kind}",
        f"   聚合算子   : {aggregation}",
        f"   网络结构   : num_layers={_g(model_config, 'num_layers')}, "
        f"hidden={_g(model_config, 'hidden_dim')}, "
        f"in_channels={feature_dim if feature_dim is not None else _g(model_config, 'in_channels')}, "
        f"dropout={_g(model_config, 'dropout')}",
        f"   文本特征   : feature_type={_g(data_config, 'feature_type')}, "
        f"feature_model={_g(model_config, 'feature_model')}, "
        f"text_model_dir={_g(data_config, 'text_model_dir') or '（空：走 HF 默认缓存）'}",
    ]
    if contrastive_config is not None:
        lines.append(
            f"   对比学习   : 启用 | α={_g(contrastive_config, 'contrastive_weight')}, "
            f"τ={_g(contrastive_config, 'temperature')}, "
            f"主父兄弟={_g(contrastive_config, 'sibling_weight')}, "
            f"辅父兄弟={_g(contrastive_config, 'sibling_secondary_weight')}, "
            f"父子={_g(contrastive_config, 'parent_weight')}, "
            f"硬负={_g(contrastive_config, 'hard_negative_weight')}"
        )
    else:
        lines.append("   对比学习   : 关闭（无对比损失）")
    lines.append(
        f"   优化配置   : lr={_g(training_config, 'lr')}, wd={_g(training_config, 'weight_decay')}, "
        f"warmup={_g(training_config, 'warmup_epochs')}, epochs={_g(training_config, 'epochs')}, "
        f"patience={_g(training_config, 'patience')}, batch_size={_g(training_config, 'batch_size')}"
    )
    if graph_data is not None:
        lines.append(
            f"   图结构     : 节点类型={list(graph_data.node_types)}, "
            f"边类型数={len(graph_data.edge_types)}"
        )
    if processor is not None:
        _csv = _g(data_config, "meddra_csv_path", "")
        lines.append(
            f"   数据规模   : 训练/验证池={_g(processor, 'num_all_nodes')} 个节点, "
            f"CSV={os.path.basename(str(_csv)) if _csv else '—'}"
        )
    lines.append(f"   时间       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n".join(lines))


def run_full_training(
    data_config: Optional[DataConfig] = None,
    model_config: Optional[ModelConfig] = None,
    training_config:  Optional[TrainingConfig] = None,
    contrastive_config:  Optional[ContrastiveConfig] = None,
    verbose: bool = True,
) -> Tuple[Optional[HeteroMedDRAEncoder],
           Optional[HeteroMedDRADataProcessor],
           Optional[Dict],
           Optional[Dict[str, pd.DataFrame]]]:
    """
    全量训练入口函数。

    参数:
        data_config (DataConfig): 数据与缓存配置
        model_config (ModelConfig): 模型架构配置
        training_config (TrainingConfig): 训练超参数配置
        contrastive_config (ContrastiveConfig): 对比学习配置
        verbose (bool): 是否打印详细信息

    返回:
        Tuple[Optional[HeteroMedDRAEncoder],
              Optional[HeteroMedDRADataProcessor],
              Optional[Dict],
              Optional[Dict[str, pd.DataFrame]]]:
            (训练好的模型, 数据处理器, 训练历史, 各层级嵌入)
    """
    # 在函数内部实例化默认值
    if data_config is None:
        data_config = DataConfig()
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()
    if contrastive_config is None:
        contrastive_config = ContrastiveConfig()

    # ---- 0. 运行目录与共享缓存 ----
    # 【目录创建放在这里】而不是 import 时：参数写错（如只写 `--jobs 2` 未给实验名）会在
    # 进入训练前报错退出，此时磁盘上不会留下任何空目录；启动器模式的父进程也不建目录。
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # 【路径类配置的环境变量覆盖】便于"一份脚本跑完所有实验"，无需为每个实验复制脚本改路径
    if os.environ.get("MEDDRA_CSV_PATH"):
        data_config.meddra_csv_path = os.environ["MEDDRA_CSV_PATH"]
    if os.environ.get("MEDDRA_TEXT_MODEL_DIR"):
        data_config.text_model_dir = os.environ["MEDDRA_TEXT_MODEL_DIR"]
    # 【先解析路径、再切目录】chdir 之后相对路径会指向新目录，所以 CSV / 文本模型目录
    #   必须在切目录**之前**就变成绝对路径，否则"相对路径配置"会静默失效（读到别的目录）。
    for _attr in ("meddra_csv_path", "text_model_dir"):
        _v = getattr(data_config, _attr, "")
        if _v and not os.path.isabs(_v):
            setattr(data_config, _attr, os.path.abspath(_v))
    # 【CSV 自动定位】配置的 CSV 不存在时（例如换机器后默认值仍是原路径），
    #   从 输出目录 / 启动目录 / 脚本目录 向上最多 3 层找 MEDDRA*.csv（取体积最大的一个）。
    #   这样"把脚本拷到任意文件夹、不带任何参数直接跑"也能自动找到数据；命中结果会写进日志。
    if not os.path.exists(str(getattr(data_config, "meddra_csv_path", "") or "")):
        _found = _find_meddra_csv((RUN_DIR, _CWD, _SCRIPT_DIR))
        if _found:
            logger.warning(
                f"⚠️ 配置的 CSV 不存在（{data_config.meddra_csv_path}）→ 已自动定位到: {_found}"
            )
            data_config.meddra_csv_path = _found
        else:
            logger.warning(
                f"⚠️ 未找到 CSV（{data_config.meddra_csv_path}）→ 请设环境变量 MEDDRA_CSV_PATH=...；"
                f" 搜索范围: 输出目录/启动目录/脚本目录 及其上 3 层"
            )
    # 切到本次输出目录：此后所有相对路径产物（PNG / 嵌入 CSV / 权重 .pth 等）都落在
    # RUN_DIR 内，实验之间互不覆盖；图/特征缓存固定用共享绝对路径，不随实验分裂。
    os.chdir(RUN_DIR)
    data_config.cache_dir = CACHE_DIR

    # 【并发锁】同一实验目录只允许一个进程：防止两个进程共写 resume.pt / 互相覆盖产物
    if not _acquire_run_lock():
        return None, None, None, None

    # 【多 seed 方差估计】环境变量覆盖（不改代码即可换训练种子、并保持划分不变）
    if os.environ.get("MEDDRA_SPLIT_SEED"):
        training_config.split_seed = int(os.environ["MEDDRA_SPLIT_SEED"])
    if os.environ.get("MEDDRA_TRAIN_SEED"):
        training_config.train_seed = int(os.environ["MEDDRA_TRAIN_SEED"])

    # 在 run_full_training 开头
    _set_seed(training_config._train_seed)
    logger.info(
        f"随机种子: 划分 split_seed={training_config._split_seed}、训练 train_seed={training_config._train_seed}"
        f"（默认均为 random_state={training_config.random_state}；可用 MEDDRA_SPLIT_SEED / MEDDRA_TRAIN_SEED 覆盖）"
    )

    

    logger.info("=" * 60)
    logger.info("🏥 MedDRA 全量训练模式")
    logger.info("=" * 60)

    if verbose:
        logger.info("训练配置:")
        logger.info(f"  数据路径: {data_config.meddra_csv_path}")
        logger.info(f"  嵌入维度: {model_config.embedding_dim}")
        logger.info(f"  训练轮数: {training_config.epochs}")
        logger.info(f"  学习率: {training_config.lr}")
        logger.info(f"  对比损失权重: {contrastive_config.contrastive_weight}")
        logger.info(f"  实验名/配置键: {EXPERIMENT}")
        logger.info(f"  实验输出目录: {RUN_DIR}")
        logger.info(f"  缓存目录: {CACHE_DIR}")
        logger.info("-" * 40)

    # ---- 1. 加载/构建图 ----
    # 【特征缓存来源标签】默认 'pubmedbert' 与既有缓存文件名一致（不触发重算）；
    # 若将来换编码器，只要 feature_type 不同就会自动用另一份缓存文件，杜绝静默复用错特征。
    GraphCacheManager.FEATURE_TAG = str(getattr(data_config, "feature_type", "pubmedbert")
                                        or "pubmedbert")
    cache_manager = GraphCacheManager(data_config.cache_dir)
    processor = HeteroMedDRADataProcessor(data_config.meddra_csv_path, cache_manager)
    graph_data, _ = processor.load_or_build_graph(
        embedding_dim=model_config.embedding_dim,
        force_rebuild=data_config.force_rebuild_cache,
        verbose=verbose,
    )
    if graph_data is None:
        logger.error("图加载失败，程序退出")
        return None, None, None, None

    if verbose:
        logger.info(f"图数据加载成功: {len(processor.node_codes):,} 个节点")

    # ---- 2. 生成或加载 PubMedBERT 特征 ----
    features_dict, feature_dim = get_pubmedbert_features(
        processor=processor,
        cache_manager=cache_manager,
        embedding_dim=model_config.embedding_dim,
        force_rebuild=data_config.force_rebuild_features,
        feature_type=data_config.feature_type,
        verbose=verbose,
        model_local_dir=data_config.text_model_dir,
    )

    if not features_dict:
        logger.error("PubMedBERT 特征生成失败")
        return None, None, None, None

    # 将特征写入图数据
    for ntype, features in features_dict.items():
        if hasattr(graph_data[ntype], "x"):
            graph_data[ntype].x = features

    # ---- 3. 创建模型 ----
    # 使用实际特征维度，而不是固定 768
    model_config.in_channels = feature_dim

    # 分类数量
    num_socs = len(processor.level_nodes.get("SOC", []))
    model = HeteroMedDRAEncoder(
        node_types=list(graph_data.node_types),
        edge_types=list(graph_data.edge_types),
        num_socs=num_socs,
        model_config=model_config,
    ).to(DEVICE)

    if verbose:
        logger.info(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 4. 训练 ----
    trainer = HeteroMedDRATrainer(
        model=model,
        data=graph_data,
        processor=processor,
        training_config=training_config,
        contrastive_config=contrastive_config,
    )

    # ---- 显式固定“模型选择（早停）协议”，保证 SAGE/GAT/NoGNN 三模型口径完全一致 ----
    # 主判据：验证准确率（Val Acc）。实测表明本任务下 macro-F1 在欠拟合阶段信噪比过低，
    #         且可能与准确率方向相反（准确率上升而 macro-F1 下降），不宜作为唯一主判据；
    #         macro-F1 仍作为报告指标，并继续参与容差带内的择优。
    #         分层级指标（HLGT/HLT/PT/LLT）同样仅用于报告与分析，不作为选择判据——
    #         因其验证样本量差异达数量级（HLGT≈27 vs LLT≈9100），等权使用会被噪声支配。
    # 次判据：主判据处于 ±es_tolerance 容差带内时，优先选 gap_average 更大的模型。
    # 保护期：前 es_min_epochs 轮只记录、不累计 patience（高学习率探索期）。
    # 停止  ：仅当“主判据未提升 且 验证损失也不再下降”时才累计 patience（避免欠拟合阶段误停）。
    # 护栏  ：相对“已选最佳权重”出现 intra↑ 且 gap 显著↓ 时，该轮按 collapse_weight 倍计入 patience。
    #         注意：不使用绝对阈值——强语义特征下 intra 可能长期处于 0.9+，绝对值高不等于塌缩。
    trainer.es_primary_metric = "acc"
    trainer.es_min_delta = 0.001          # 主指标“严格提升”阈值（真实历史回放校准：0.002 会在平台期提前 15 轮停止选中次优权重；≥0.005 严重欠拟合）
    trainer.es_tolerance = 0.005          # 容差带（“同水平”判定）
    trainer.es_gap_eps = 1e-4             #  Minimum gap improvement required to select a better model inside the tolerance band
    trainer.es_min_epochs = 60            # 早停保护期：前 60 轮只记录、不累计 patience（≈200轮的30%）
    trainer.es_loss_min_delta = 1e-4      #  Threshold for deciding that the validation loss is "still decreasing"
    trainer.collapse_weight = 3           # 塌缩轮等价计为几轮无进展（15/3 = 5 轮触发）
    trainer.collapse_gap_ratio = 0.1      # gap 相对参考值的最小下降比例（塌缩判定）
    logger.info(
        f"🧭 模型选择协议: 主判据={trainer.es_primary_metric}, "
        f"min_delta={trainer.es_min_delta}, 容差={trainer.es_tolerance}, "
        f"保护期={trainer.es_min_epochs}轮, "
        f"损失阈值={trainer.es_loss_min_delta}, "
        f"塌缩加权=×{trainer.collapse_weight}"
        f"（等效反应 {max(1, training_config.patience // max(1, trainer.collapse_weight))} 轮）"
    )

    # ---- 运行自证横幅（A2）：让任何日志都能自证是哪个模型，不依赖文件名 ----
    log_run_banner(
        model_kind="SAGE（本文主模型：HeteroConv + SAGEConv）",
        aggregation="均值聚合（SAGEConv root_weight=True，显式保留节点自身特征）",
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        contrastive_config=contrastive_config,
        graph_data=graph_data,
        processor=processor,
        verbose=verbose,
    )

    trained_model, history = trainer.train(verbose=verbose)

    # ---- 生成实验标识，避免消融实验互相覆盖（v7.1 修复）----
    # 说明：所有输出文件都带此 tag，确保 SAGE-full / SAGE-noCL / τ 消融 / 权重消融 / 随机特征
    #       等不同实验的产物不互相覆盖，便于后续论文表格填数。
    # 权重部分（保留）
    weight_tag = (
        f"sib{contrastive_config.sibling_weight}"
        f"sib2{contrastive_config.sibling_secondary_weight}"
        f"par{contrastive_config.parent_weight}"
        f"hard{contrastive_config.hard_negative_weight}"
    )
    # 开关部分（新增）
    switch_tag = (
        f"psib{int(contrastive_config.use_primary_sibling)}"
        f"ssib{int(contrastive_config.use_secondary_sibling)}"
        f"par{int(contrastive_config.use_parent)}"
        f"hneg{int(contrastive_config.use_hard_negative)}"
        f"excl{int(contrastive_config.use_mutual_exclusion)}"
    )
    experiment_tag = (
        f"{data_config.feature_type}_"
        f"{'cl' if training_config.use_contrastive else 'noCL'}_"
        f"tau{contrastive_config.temperature}_"
        f"{weight_tag}_{switch_tag}"
    )
    logger.info(f"📌 实验标识: {experiment_tag}")

    # 【隔离自检】最终产物一律写进本实验目录：PNG / CSV / 权重 / checkpoint 都用相对路径保存，
    # 一旦 cwd 不是 RUN_DIR 就会落错地方（实验之间互相污染）→ 这里强制纠正回来。
    if os.path.abspath(os.getcwd()) != os.path.abspath(RUN_DIR):
        logger.warning(f"⚠️ 当前工作目录不是实验目录（{os.getcwd()}）→ 已纠正为 {RUN_DIR}")
        os.chdir(RUN_DIR)

    # 保存训练历史（带 tag；写入本实验目录，与缓存目录分离）
    cache_manager.save_train_history(
        data_config.meddra_csv_path,
        model_config.embedding_dim,
        history,
        tag=experiment_tag,
        out_dir=RUN_DIR,
    )

    # ---- 5. 可视化与导出 ----

    level_embeddings = trainer.generate_embeddings()
    if level_embeddings:
        ModelExporter.save_embeddings(
            level_embeddings,
            output_prefix=f"meddra_full_embeddings_{experiment_tag}",
        )
        ModelExporter.save_model(
            trained_model,
            f"best_meddra_encoder_{experiment_tag}.pth",
        )
        # 【新增】保存完整 checkpoint
        ModelExporter.save_checkpoint(
            model=trained_model,
            processor=processor,
            model_config=model_config,
            training_config=training_config,
            contrastive_config=contrastive_config,
            best_metrics=trainer.best_metrics,
            filepath=f"best_checkpoint_{experiment_tag}.pt",
            test_metrics=getattr(trainer, "test_metrics", None) or None,
        )

    # 训练历史 + 降维可视化：**全部容错**（绘图失败只告警；权重/指标/嵌入此前已落盘）
    _safe_visual(Visualizer.plot_training_history, history,
                 save_path=f"training_history_{experiment_tag}.png")

    _safe_visual(Visualizer.visualize_embeddings_by_all_soc, level_embeddings,
                 save_path=f"visualize_embeddings_all_soc_{experiment_tag}.png",
                 processor=processor)

    _safe_visual(Visualizer.visualize_embeddings_by_part_soc, level_embeddings,
                 save_path=f"visualize_embeddings_part_soc_{experiment_tag}.png",
                 processor=processor)

    _safe_visual(Visualizer.visualize_soc_confusion, level_embeddings,
                 processor=processor,
                 save_path=f"visualize_soc_confusion_{experiment_tag}.png")

    # ---- 6. 构建记忆库 ----
    # [改进] 构建记忆库，传入采样策略
    if data_config.enable_memory_bank:
        memory_bank = build_memory_bank(
            trainer,
            k_per_soc=data_config.memory_k_per_soc,
            output_file=f"memory_bank_{experiment_tag}.pkl",
            sampling_strategy=data_config.memory_sampling_strategy,
            seed=training_config.random_state,
        )
        logger.info(f"📦 记忆库大小: {len(memory_bank)} 个样本")
    else:
        logger.info(
            "⏭️ 跳过记忆库构建（enable_memory_bank=False）"
            "：该产物仅为增量训练预留，当前训练/评估流程均不消费"
        )

    device_manager.clear_memory()

    logger.info("\n🎉 全量训练完成！")
    bm = trainer.best_metrics
    logger.info(
        f"📊 选中模型 (Epoch {trainer.best_epoch+1}, "
        f"主判据={bm.get('es_primary', 'acc')}, 依据={bm.get('es_selected_by', 'strict')}): "
        f"Acc={bm.get('accuracy', 0):.4f}, Macro-F1={bm.get('macro_f1', 0):.4f}, "
        f"Weighted-F1={bm.get('weighted_f1', 0):.4f}"
    )
    logger.info(
        f"📊 历史最高验证准确率: {trainer.best_val_acc:.4f} | "
        f"选中模型 Gap(几何): {bm.get('gap_average', 0):.4f}"
    )
    logger.info(
        f"🔎 结构线索诊断（选中权重）: 主父∈train acc="
        f"{bm.get('diag_parent_in_train_acc', 0):.4f} (n={bm.get('diag_parent_in_train_n', 0)}) vs "
        f"主父∉train acc={bm.get('diag_parent_out_train_acc', 0):.4f} "
        f"(n={bm.get('diag_parent_out_train_n', 0)}), Gap={bm.get('diag_parent_gap', 0):+.4f}"
    )
    if bm:
        logger.info(
            f"📊 Per-level Acc: HLGT={bm.get('hlgt_acc', 0):.4f}, "
            f"HLT={bm.get('hlt_acc', 0):.4f}, "
            f"PT={bm.get('pt_acc', 0):.4f}, "
            f"LLT={bm.get('llt_acc', 0):.4f}"
        )
        logger.info(
            f"📊 Per-level Macro-F1: HLGT={bm.get('hlgt_macro_f1', 0):.4f}, "
            f"HLT={bm.get('hlt_macro_f1', 0):.4f}, "
            f"PT={bm.get('pt_macro_f1', 0):.4f}, "
            f"LLT={bm.get('llt_macro_f1', 0):.4f}"
        )
    # 记忆库大小已在构建分支内打印（默认关闭时不构建）
    logger.info(f"📌 实验标识: {experiment_tag}")

    return trained_model, processor, history, level_embeddings


# =============================================================================
# 13. 主入口
# =============================================================================

# =============================================================================
# 命令行启动器：python <本脚本> V1 V2 C3 [--jobs N] [--dry-run]
# -----------------------------------------------------------------------------
# 机制（命令行模式只当"启动器"，自己不训练）：
#   ① 为每个实验名创建/复用目录 <结果根>/<实验名>/（已存在则复用 → 断点续训可直接接着跑）
#   ② 把本脚本复制一份进去（版本留档：该目录里的脚本就是产生该结果的代码）
#   ③ 用**独立子进程**在各自目录里运行它（子进程互不共享状态，避免显存/缓存串味）
#   ④ 用环境变量把实验名与输出目录传给子进程，子进程无需再解析参数
# 结果根目录 = $MEDDRA_OUTPUT_ROOT，未设置则取**当前工作目录**。
# 显存提醒：--jobs>1 时 SAGE/NoGNN 约 3–4 GB/个；GAT 约 11–12 GB，建议 GAT 独占。
# =============================================================================
_LOCK_PATH = ""          # 本次运行持有的锁文件路径（atexit 时释放）


def _release_run_lock() -> None:
    """释放本次运行持有的锁（正常结束或异常退出时由 atexit 调用）。"""
    global _LOCK_PATH
    if _LOCK_PATH and os.path.exists(_LOCK_PATH):
        try:
            with open(_LOCK_PATH, encoding="utf-8") as f:
                if f.readline().strip() == str(os.getpid()):
                    os.remove(_LOCK_PATH)
        except Exception:
            pass
    _LOCK_PATH = ""


def _pid_alive(pid: int) -> bool:
    """判断进程是否还活着 —— **跨平台且不发送任何信号**。

    ⚠️ 不能用 `os.kill(pid, 0)`：在 Windows 上它并不是"存在性检查"，而是会
    直接 TerminateProcess 把目标进程杀掉（本文件早期版本踩过这个坑）。
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            return True                     # 判断不了就保守当作"活着"（宁可不启动，也不双写）
    try:
        os.kill(pid, 0)                     # POSIX 下这是标准的存在性检查（不发信号）
        return True
    except ProcessLookupError:              # 进程确实不存在 → 陈旧锁
        return False
    except PermissionError:                 # 存在但属于别的用户（多用户服务器常见）→ 保守当作活着
        return True
    except OSError:                         # 其它异常也保守当作活着：宁可不启动，也不要双写
        return True


def _acquire_run_lock() -> bool:
    """同一实验目录只允许一个进程运行 —— 防止两个进程共写 `resume.pt` 与互相覆盖产物。

    锁文件带 PID：若锁里的 PID 已不存在（进程被杀/断电），视为陈旧锁并接管。
    设环境变量 `MEDDRA_NO_LOCK=1` 可跳过该检查。返回 True 表示已持有锁。
    """
    global _LOCK_PATH
    if os.environ.get("MEDDRA_NO_LOCK"):
        return True
    path = os.path.join(RUN_DIR, ".run.lock")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                _pid = int(f.readline().strip() or "0")
        except Exception:
            _pid = 0
        if _pid > 0 and _pid != os.getpid():
            _alive = _pid_alive(_pid)
            if _alive:
                logger.error(f"❌ 该实验目录已有进程在运行（PID {_pid}）→ 为避免两个进程同时写 "
                             f"resume.pt 与结果文件，本次直接退出。")
                logger.error(f"   确认没有进程在跑时可删除锁文件后重试: {path}")
                return False
            logger.warning(f"发现陈旧锁（PID {_pid} 已不存在）→ 接管: {path}")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _LOCK_PATH = path
        import atexit
        atexit.register(_release_run_lock)
    except Exception as e:
        logger.warning(f"写锁文件失败（不影响训练）: {e}")
    return True


def _resolve_experiment(name: str, registry: dict, default_key: str) -> str:
    """把"文件夹名 / 环境变量里的实验名"解析成注册表里的配置键。

    解析顺序：
      ① 精确命中；
      ② **等价命名互换**：`IND-` / `NoGNN-` 前缀的有无视为同一个实验
         （如 `GAT` ↔ `IND-GAT`、`PubMedBert` ↔ `NoGNN-PubMedBert`）；
      ③ 都不中 → 用默认项并**大声告警**。
    这样"把脚本放进任意文件夹、直接 `python 脚本.py` 跑"永远能跑起来，
    同时把"实际用了哪个配置"写进日志（而不是抛 KeyError 只留一个空文件夹）。
    """
    if name in registry:
        return name
    _cands = [f"IND-{name}", f"NoGNN-{name}"]
    for _p in ("IND-", "NoGNN-", "NoGNN"):
        if name.startswith(_p):
            _base = name[len(_p):]
            _cands += [_base, f"IND-{_base}", f"NoGNN-{_base}"]
    for _c in _cands:
        if _c in registry:
            logger.info(f"实验名 {name!r} → 按注册表中的 {_c!r} 解析（同一实验的两种写法）")
            return _c
    logger.error(
        f"⚠️ 未知实验名 {name!r}（本文件可选: {sorted(registry)}）→ 本次按默认配置 "
        f"{default_key!r} 运行，结果仍写入 {RUN_DIR}；如需其它配置请设 MEDDRA_EXPERIMENT=<实验名>，"
        f"或把脚本放进名称匹配的文件夹。"
    )
    return default_key


def _parse_cli(argv):
    """解析 `V1 V2 C3 --jobs 2 --dry-run`，返回 (实验名列表, 并发数, 是否 dry-run)。"""
    names, jobs, dry = [], 1, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--dry-run", "-n"):
            dry = True
            i += 1
        elif a in ("--jobs", "-j"):
            if i + 1 >= len(argv):
                raise SystemExit("--jobs 需要一个整数参数，例如 --jobs 2")
            jobs = max(1, int(argv[i + 1]))
            i += 2
        elif a.startswith("--jobs="):
            jobs = max(1, int(a.split("=", 1)[1]))
            i += 1
        else:
            names.append(a)
            i += 1
    return names, jobs, dry


def _launch_experiments(names, registry, jobs=1, dry_run=False) -> int:
    """把每个实验跑在自己的文件夹里；返回 0 表示全部成功。"""
    import subprocess

    root = os.environ.get("MEDDRA_OUTPUT_ROOT") or os.getcwd()
    src = _SCRIPT_PATH
    if not src:
        logger.error("当前环境没有脚本文件路径（如 Jupyter / Kaggle 单元格），无法自动复制脚本。")
        logger.error("请改用：MEDDRA_EXPERIMENT=<实验名> MEDDRA_OUTPUT_ROOT=<结果根> python <脚本>.py")
        logger.error("或把本脚本保存成 .py 文件后再用：python <脚本>.py V1 V2 C3 [--jobs N]")
        return 2
    script_name = os.path.basename(src)

    unknown = [n for n in names if n not in registry]
    if unknown:
        logger.error(f"未知实验名: {unknown}")
        logger.error(f"本脚本可选: {sorted(registry)}")
        return 2

    # 目录可用性预检：目标路径被同名**文件**占用时立刻报错（而不是跑到一半才崩）
    for n in names:
        _d = os.path.join(root, n)
        if os.path.exists(_d) and not os.path.isdir(_d):
            logger.error(f"❌ 结果路径被同名文件占用，无法创建实验目录: {_d}")
            return 2

    logger.info("=" * 60)
    logger.info(f"🚀 批量运行 {len(names)} 个实验（{'串行' if jobs <= 1 else f'最多 {jobs} 并发'}）")
    logger.info(f"   结果根目录: {root}")
    for n in names:
        _ov = registry.get(n) or {}
        _has_ckpt = os.path.exists(os.path.join(root, n, f"resume_{_MODEL_TAG}.pt"))
        logger.info(
            f"   · {n:16s} → {os.path.join(root, n)}"
            f"   配置覆盖={_ov if _ov else '默认（完全体）'}"
            f"{'   ♻️ 检测到断点，将续训' if _has_ckpt else ''}"
        )
    logger.info("=" * 60)
    if dry_run:
        logger.info("--dry-run：仅打印计划，未实际运行")
        return 0

    def _prepare(n):
        d = os.path.join(root, n)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, script_name)
        if os.path.abspath(dst) != src:          # 防止把脚本复制到它自己
            shutil.copy2(src, dst)
        # 【溯源】记录"这份结果用的是哪版代码"：源路径 + 内容 md5 + 复制时间 + 版本号。
        # 脚本副本本身可读，但无法自证与源一致；这份 JSON 可以。
        try:
            with open(src, "rb") as _f:
                _md5 = hashlib.md5(_f.read()).hexdigest()
            with open(os.path.join(d, "RUN_META.json"), "w", encoding="utf-8") as _f:
                json.dump({
                    "experiment": n,
                    "run_dir": d,
                    "script_source": src,
                    "script_name": script_name,
                    "script_md5": _md5,
                    "copied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output_root": root,
                    "python": sys.version.split()[0],
                    "build_hierarchy_version": BUILD_HIERARCHY_VERSION,
                    "cache_version": GraphCacheManager.CACHE_VERSION,
                    "argv": list(sys.argv),
                }, _f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"写 RUN_META.json 失败（不影响运行）: {e}")
        return d, dst

    def _env(n, d):
        e = dict(os.environ)
        e["MEDDRA_EXPERIMENT"] = n               # 显式指定实验名（优先于脚本内默认值）
        e["MEDDRA_RUN_DIR"] = d                  # 显式指定输出目录
        e.setdefault("MEDDRA_OUTPUT_ROOT", root)
        return e

    fails = []
    if jobs <= 1:
        for n in names:
            d, dst = _prepare(n)
            logger.info(f"▶️ 开始 {n}（工作目录 {d}）")
            rc = subprocess.run([sys.executable, dst], cwd=d, env=_env(n, d)).returncode
            logger.info(f"{'✅' if rc == 0 else '❌'} {n} 结束（返回码 {rc}）")
            if rc != 0:
                fails.append(n)
    else:
        pending = list(names)
        running = {}                             # 实验名 -> Popen
        while pending or running:
            while pending and len(running) < jobs:
                n = pending.pop(0)
                d, dst = _prepare(n)
                running[n] = subprocess.Popen([sys.executable, dst], cwd=d, env=_env(n, d))
                logger.info(f"▶️ 启动 {n}（并行 {len(running)}/{jobs}）")
            for n, p in list(running.items()):
                if p.poll() is not None:
                    logger.info(f"{'✅' if p.returncode == 0 else '❌'} {n} 结束（返回码 {p.returncode}）")
                    if p.returncode != 0:
                        fails.append(n)
                    del running[n]
            if running:
                time.sleep(5)

    logger.info("=" * 60)
    logger.info(f"批量运行结束：成功 {len(names) - len(fails)}/{len(names)}"
                + (f"；失败 {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    # Direct call to run_full_training; no registry / no CLI; config hardcoded for the C7 seed repeat.
    _data_cfg = DataConfig()
    _train_cfg = TrainingConfig()
    _contr_cfg = ContrastiveConfig()
    _contr_cfg.use_primary_sibling = False
    _contr_cfg.use_parent = False
    _train_cfg.split_seed = 42
    _train_cfg.train_seed = 43
    run_full_training(
        data_config=_data_cfg,
        model_config=ModelConfig(),
        training_config=_train_cfg,
        contrastive_config=_contr_cfg,
    )
    print(f"C7-seed43 done. Artifacts in: {RUN_DIR}")
