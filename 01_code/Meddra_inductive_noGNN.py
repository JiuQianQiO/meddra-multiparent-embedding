# =============================================================================
# 文件: meddra_noGNN_trainer.py
# 功能: MedDRA 纯 MedBERT 微调（无 GNN，无对比学习）
# 版本: 7.1-noGNN
# 说明:
#   1. 只使用 MedBERT 特征 + 线性分类器进行 SOC 分类
#   2. 删除所有图卷积、对比损失、投影头
#   3. 保留缓存、可视化、早停（基于验证准确率）
#   4. 验证时计算各层级（HLGT/HLT/PT/LLT）的簇内余弦相似度并记录
#   5. SOC 标签逻辑与 SAGE/GAT 完全对齐（v7.2 分层策略，三文件一致）：
#      PT 使用官方 PT_SOC_CODE、LLT 继承父 PT、HLT/HLGT 由子节点投票聚合（众数）；
#      PRIMARY_SOC_FG 仅用于确定主父。
#      注意：早期版本用于“父节点继承回填”的 _infer_missing_soc_labels 已废弃，
#      当前实现**不再调用**（定义保留仅为兼容旧版，见方法处的 [已废弃] 标注）。
# =============================================================================

import os
import sys
import shutil
import json
import pickle
import hashlib
import time
import gc
import random                     
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
from torch_geometric.nn import Linear
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =============================================================================
# 【运行目录】每实验一个文件夹 + 共享缓存目录
# -----------------------------------------------------------------------------
# 设计目标：一次运行只写一个文件夹，实验之间互不覆盖，便于归档与论文取数。
#   · RUN_DIR   —— 本次实验的**唯一输出目录**：日志 / 训练历史 JSON / checkpoint /
#                  权重 .pth / 嵌入 CSV / 各张 PNG 全部落在里面。
#   · CACHE_DIR —— 图缓存与文本特征缓存，**跨实验共享**，不随实验变化。
#                  原因：缓存只与"CSV / 维度 / 编码器"有关，与实验配置无关；noGNN 三种编码器
#                        的特征缓存文件名已含 feature_model，彼此不会覆盖。
# 目录选择优先级（高 → 低）：
#   ① 环境变量 MEDDRA_RUN_DIR                        直接指定输出目录（实验名取该目录名）
#   ② 环境变量 MEDDRA_OUTPUT_ROOT + MEDDRA_EXPERIMENT  拼出"输出根 / 实验名"
#   ③ 默认 = **当前工作目录**：不指定任何路径 → 日志 / 训练历史 / 断点 / 权重 / 嵌入 CSV / PNG
#      以及缓存，全部落在"当前文件夹"（与脚本叫什么名字、放在哪里、有没有参数都无关）。
#   缓存目录：MEDDRA_CACHE_DIR > <输出目录>/meddra_cache
#      （想跨实验共用一份缓存 → 显式设 `MEDDRA_CACHE_DIR=<公共目录>`）
# 用法：cd 到某文件夹 → python <任意名字的脚本>.py              # 全部结果落这里
#       cd 到某文件夹 → python <脚本>.py PubMedBert --jobs 1    # 启动器：自动建 PubMedBert/
# =============================================================================
# 【兼容 Kaggle / Jupyter】单元格里没有 __file__ → 安全回退成"无脚本路径"；
# 那种环境下请用环境变量指定实验名与输出目录（MEDDRA_EXPERIMENT / MEDDRA_OUTPUT_ROOT）。
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
        # 必须真实存在，且排除 ipykernel 启动器（它是内核的脚本，不是用户的实验脚本）
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
                    "MEDDRA_TRAIN_SEED", "MEDDRA_SPLIT_SEED", "MEDDRA_IND_SELECT", "MEDDRA_KEEP_EDGES") if k in os.environ},
            }, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_MODEL_TAG   = "IND-noGNN"  # 本文件身份（用于断点文件名：不同模型在同一目录不会互相覆盖）
_SCRIPT_PATH = _safe_script_path()
_SCRIPT_DIR  = os.path.dirname(_SCRIPT_PATH) if _SCRIPT_PATH else ""
_CWD         = os.getcwd()

if os.environ.get("MEDDRA_RUN_DIR"):
    # ① 显式指定输出目录：实验名取该目录名（目录名才是实验身份的唯一来源）
    RUN_DIR     = os.path.abspath(os.environ["MEDDRA_RUN_DIR"])
    EXPERIMENT  = os.environ.get("MEDDRA_EXPERIMENT") or _label(RUN_DIR)
    OUTPUT_ROOT = os.path.dirname(RUN_DIR)
elif os.environ.get("MEDDRA_EXPERIMENT"):
    # ② 只给了实验名：输出根 = MEDDRA_OUTPUT_ROOT > 当前目录
    EXPERIMENT  = os.environ["MEDDRA_EXPERIMENT"]
    _root       = (os.path.abspath(os.environ["MEDDRA_OUTPUT_ROOT"])
                   if os.environ.get("MEDDRA_OUTPUT_ROOT") else _CWD)
    RUN_DIR     = os.path.join(_root, EXPERIMENT)
else:
    # ③ 【默认】当前工作目录：不指定任何路径 → 日志/结果/缓存全部落在"当前文件夹"
    #    （与脚本叫什么名字、放在哪个文件夹、有没有参数都无关）
    RUN_DIR     = _CWD
    EXPERIMENT  = _label(RUN_DIR)

OUTPUT_ROOT = os.path.dirname(RUN_DIR)     # 仅用于日志展示与旧配置兼容
LABEL       = _label(EXPERIMENT)           # 文件名安全标签（实验名可能来自环境变量，可能含非法字符）
CACHE_DIR   = os.environ.get("MEDDRA_CACHE_DIR") or os.path.join(
    RUN_DIR, "meddra_cache")               # 缓存默认放当前目录下的 meddra_cache（自包含）
# 【启动器模式】命令行里带了实验名 → 本进程只负责"建目录 + 复制脚本 + 派生子进程"，
# 自己不训练，因此不创建运行目录、也不写日志文件（避免留下空目录/空日志）。
# 启动器只在"以 .py 文件运行"时启用：notebook 里 sys.argv 是内核参数（含路径），不能当实验名
_CLI_NAMES = [a for a in sys.argv[1:]
              if _SCRIPT_PATH and a and not a.startswith("-")
              and "/" not in a and "\\" not in a and "." not in a
              and not a.isdigit()]        # 排除 `--jobs 2` 里的数字：别把作业数当成实验名
_LAUNCH_MODE = bool(_CLI_NAMES)

if not _LAUNCH_MODE:
    os.makedirs(RUN_DIR, exist_ok=True)
    _write_run_meta()      # 直接运行也留一份溯源记录（脚本路径 / md5 / 参数 / 解析出的目录）
# 缓存目录**延迟到真正训练时**再创建（见 run_full_training）：这样参数写错（如只写 `--jobs 2`
# 没给实验名）在报错退出时不会在磁盘上留下任何空目录；启动器模式的父进程也不建任何目录。

# 【断点续训】续训文件放在本次输出目录；文件名带模型标签
# （同一目录偶尔跑两个模型时，各自的断点不会互相覆盖）
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


# 设置日志：控制台 + 本实验目录下的文件（双写）
# 【坑位说明】
#   1) 必须用 handlers=[...]：只写 basicConfig(filename=...) 会让 FileHandler 取代默认 StreamHandler，
#      控制台将不再输出；
#   2) force=True：transformers/torch_geometric 等库可能已先给 root 挂 handler，此时 basicConfig
#      会静默失效（不报错、也不生成文件）；
#   3) encoding='utf-8'：Windows 默认 GBK，日志中的中文/emoji 会 UnicodeEncodeError；
#   4) 文件名含实验名 + 时间戳：既区分实验，也避免同实验重跑覆盖旧日志；
#   5) tqdm 进度条绕过 logging 直写 stderr，只进控制台、不进文件；print/异常 traceback 同样不受 FileHandler 捕获。
LOG_FILE = os.path.join(RUN_DIR, f"run_{_MODEL_TAG}_{LABEL}_{time.strftime('%Y%m%d_%H%M%S')}.log")
_handlers = [logging.StreamHandler()]                       # 控制台（默认 stderr，与原先行为一致）
if not _LAUNCH_MODE:
    _handlers.append(
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')   # 本实验目录下的日志文件
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
# 层级/标签构建版本 —— 单一来源（single source of truth）
# -----------------------------------------------------------------------------
# 该常量同时用于三处：
#   ① 缓存键（隔离不同版本的图/特征缓存）；
#   ② 缓存元数据与 checkpoint 中的版本标记；
#   ③ 加载时的「版本差异检测」——不一致即判定缓存/断点失效并重建。
# 【规则】只要改动会影响 hierarchy_map / node_soc_map / _primary_parent / 图结构或边，
#        必须升号；否则旧缓存会被静默复用，标签与主父停留在旧口径。
# 说明：各训练脚本的缓存键保留自身后缀（-SAGE 无后缀 / -GAT / -noGNN）以避免互相覆盖。
# =============================================================================
BUILD_HIERARCHY_VERSION = "v7.2"


# 【新增】全局随机种子控制（与 SAGE 版本一致）
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resume_sanitize(obj):
    """把 numpy 标量/数组递归换成原生类型。

    用途：断点续训文件要能被 `torch.load(weights_only=True)` 读回（torch>=2.6 的默认值），
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


# =============================================================================
# 【归纳式（子树留出）协议】—— 本文件专属，与转导式主实验完全隔离
# -----------------------------------------------------------------------------
# 动机：V3（纯随机特征 + 图结构）在转导设定下已达 0.9762 → 需要量化"结构线索"的贡献。
#       noGNN 不用图、也不做消息传递，故它**本身就没有结构线索可利用**；本文件的作用是
#       提供一个**与 SAGE/GAT 完全同划分**的无图基线：
#         · 验证集 = 整棵留出 HLT 子树（HLT + PT + LLT）；
#         · 训练集 = 其余节点（比转导式更小）。
#       这样"SAGE 归纳式 vs NoGNN 归纳式"就是同数据、同划分下的公平对比。
# 说明：因无消息传递，**无需**对边做掩码（遮不遮都不影响它的前向计算）。
# =============================================================================
INDUCTIVE_MODE = True          # 本文件恒为归纳式（专用文件，无需开关）
INDUCTIVE_HLT_RATIO = 0.10     # 留出的 HLT 子树比例（与 SAGE/GAT 归纳式文件保持一致）
INDUCTIVE_SEED = 42            # 留出选择的随机种子（三模型必须一致，否则划分不可比）
# 【v1/v2 协议开关】本文件**恒为 no-op**：NoGNN 没有任何消息传递，遮不遮边都不影响其前向
# （见上方说明）。保留该常量只为与 SAGE/GAT 归纳式文件保持接口一致，并参与断点指纹，
# 使"同一留出集、不同协议"的断点不会互相复用。
#   False = v1（论文 4.4.5 所用）；True = v2（保留层级边，仅遮蔽标签）。
INDUCTIVE_KEEP_EDGES = False
# 【JBI 投稿】协议切换已接入实验注册表：IND-*-MASK（本常量=False）/ IND-*-KEEP（=True），
# 也可用环境变量 MEDDRA_KEEP_EDGES 强制指定（优先级最高）；手工改本常量的方式已废弃，
# 以免与配置名 IND-V2（去对比学习）再次撞名（500/Inductive/V2/SAGE 那次即因此跑成 noCL）。
# ⚠️ 【命名撞车提醒】"协议 v2（保留层级边）"与配置名「V2 / IND-V2（去对比学习）」**不是同一件事**：
#   若把本脚本放进名为 V2 的文件夹，实验名解析会按等价命名互换把它当成 IND-V2 → 静默关掉对比学习
#   （500/Inductive/V2/SAGE 那次就是这样，checkpoint 名里出现 noCL）。
#   要跑"协议 v2"：把上面的常量改为 True，并用 MEDDRA_EXPERIMENT 显式指定实验名（如 IND-V1），
#   或把脚本放进非注册表名的文件夹（如 KEEP-EDGES）。下面在启用时给出显式告警。
if INDUCTIVE_KEEP_EDGES:
    logger.warning(
        "⚠️ 已启用【协议 v2：保留留出节点的层级边】——注意这与配置名 V2 / IND-V2（去对比学习）"
        "不是同一件事；若同时要保持完全体配置，请设 MEDDRA_EXPERIMENT=IND-V1，"
        "或把脚本放进非注册表名的文件夹（如 KEEP-EDGES）。当前 EXPERIMENT=%s", EXPERIMENT
    )


# =============================================================================
# 【2026-10-29 审稿补充】归纳式模型选择集协议 + 训练种子解耦
# -----------------------------------------------------------------------------
# 第一轮（默认 holdout）：验证集 = 留出集 → 早停与选权重都在留出集上完成，报告的
#   Held-out Acc 是"在被评估集合上挑出的最好一轮"，带有选择乐观偏差。
# disjoint（环境变量 MEDDRA_IND_SELECT=disjoint）：
#   · 模型选择集 = 从**非留出**节点按 SOC 分层抽取 val_ratio（划分种子 split_seed，默认 42）；
#   · 留出集只在选定权重上评估一次 → holdout_metrics_<模型标签>.json /
#     holdout_predictions_<模型标签>.json；
#   · 留出集本身不变（仍由 INDUCTIVE_SEED=42 选定），可与第一轮逐成员核对。
# 训练种子（MEDDRA_TRAIN_SEED）只影响模型初始化、逐轮打乱与 dropout；
# 划分种子（MEDDRA_SPLIT_SEED）只影响模型选择集的抽样。二者都不设时与第一轮逐位一致。
# =============================================================================
_IND_SELECT_RAW = str(os.environ.get("MEDDRA_IND_SELECT", "holdout") or "holdout").strip().lower()
if _IND_SELECT_RAW not in ("holdout", "disjoint"):
    raise SystemExit("MEDDRA_IND_SELECT 只能是 holdout 或 disjoint，当前为 %r" % _IND_SELECT_RAW)
INDUCTIVE_SELECT_DISJOINT = _IND_SELECT_RAW == "disjoint"


def _inductive_disjoint_split(all_nodes, all_soc_labels, holdout, val_ratio, split_seed):
    """disjoint 协议的三分：返回 (train_idx, select_idx, holdout_idx)。

    · holdout_idx：code ∈ 留出集的全部节点实例（与第一轮的验证集逐个相同）；
    · select_idx ：非留出节点内按 SOC 分层抽样（与转导式 val 的抽样规则相同：
                   每个 SOC 取 max(1, min(int(n*val_ratio), n//2))，n≤2 时取第 1 个）；
    · train_idx  ：其余非留出节点（含无 SOC 标签的兜底节点，与第一轮一致）。
    纯函数，不依赖 torch，便于离线单测。
    """
    holdout = set(int(c) for c in holdout)
    holdout_idx = [i for i, c in enumerate(all_nodes) if int(c) in holdout]
    by_soc = defaultdict(list)
    unlabeled = []
    for i, (c, lab) in enumerate(zip(all_nodes, all_soc_labels)):
        if int(c) in holdout:
            continue
        if lab >= 0:
            by_soc[lab].append(i)
        else:
            unlabeled.append(i)
    rng = np.random.RandomState(int(split_seed))
    train_idx, select_idx = [], []
    for lab in sorted(by_soc):
        indices = by_soc[lab]
        n_val = max(1, min(int(len(indices) * val_ratio), len(indices) // 2))
        if len(indices) <= 2:
            val_sel, train_sel = list(indices[:1]), list(indices[1:])
        else:
            val_sel = [int(x) for x in rng.choice(indices, n_val, replace=False)]
            _vs = set(val_sel)
            train_sel = [i for i in indices if i not in _vs]
        select_idx.extend(val_sel)
        train_idx.extend(train_sel)
    train_idx.extend(unlabeled)
    return sorted(train_idx), sorted(select_idx), holdout_idx


def select_inductive_holdout(processor, ratio: float = INDUCTIVE_HLT_RATIO,
                             seed: int = INDUCTIVE_SEED) -> Set[int]:
    """按固定种子挑选若干**整棵 HLT 子树**作为留出集，返回被留出的节点 code 集合。"""
    hlt_nodes = sorted(int(c) for c in processor.level_nodes.get("HLT", []))
    if not hlt_nodes:
        raise RuntimeError("归纳式：未找到 HLT 节点，无法构造留出子树")
    rng = np.random.RandomState(seed)
    n_hlt = max(1, int(round(len(hlt_nodes) * ratio)))
    chosen = [int(c) for c in rng.choice(hlt_nodes, n_hlt, replace=False)]
    holdout: Set[int] = set()
    for hlt in chosen:
        stack = [hlt]
        while stack:
            c = stack.pop()
            if c in holdout:
                continue
            holdout.add(c)
            stack.extend(int(x) for x in processor.hierarchy_map.get(c, {}).get("children", []))
    return holdout

# =============================================================================
# 0. 配置类（仅保留必要配置）
# =============================================================================

@dataclass
class ModelConfig:
    embedding_dim: int = 256          # 最终嵌入维度
    hidden_dim: int = 256             # 投影层维度
    dropout: float = 0.3              # Dropout 比率
    in_channels: int = 768            # PubMedBERT 输出维度，由数据决定
    feature_model: str = "PubMedBERT" # PubMedBERT / SapBERT / CODER

@dataclass
class TrainingConfig:
    epochs: int = 500         # 初始上限，同时是余弦退火周期 T_max；与 SAGE/GAT 保持一致
    lr: float = 1e-3
    patience: int = 15                # 早停耐心（基于验证准确率）
    batch_size: int = 512
    weight_decay: float = 5e-5
    warmup_epochs: int = 3
    val_ratio: float = 0.1
    random_state: int = 42  # 固定写死
    # 【2026-10-29 多 seed】划分种子与训练种子解耦（默认 None = 沿用 random_state，与第一轮逐位一致）：
    #   · split_seed —— 只在 disjoint 协议下决定模型选择集的抽样；
    #   · train_seed —— 决定模型初始化、逐轮打乱与 dropout（`train_seed*1000 + epoch` 逐轮播种）。
    #   亦可用环境变量覆盖：MEDDRA_SPLIT_SEED / MEDDRA_TRAIN_SEED。
    split_seed: Optional[int] = None
    train_seed: Optional[int] = None

    @property
    def _split_seed(self) -> int:
        return self.random_state if self.split_seed is None else int(self.split_seed)

    @property
    def _train_seed(self) -> int:
        return self.random_state if self.train_seed is None else int(self.train_seed)
    # 【断点续训】默认开启：若运行目录下存在 resume_<模型标签>.pt（且配置/版本/图规模一致），
    #   自动从下一轮继续；训练正常结束（含早停/NaN 退出）后该文件会被自动删除。
    #   关闭方式：resume=False 或设环境变量 MEDDRA_NO_RESUME=1。
    resume: bool = True
    resume_every: int = 10  # 每多少轮落一次续训文件（0 = 不落盘，仅关闭保存）
    # 【断点续训·读取来源】留空 = 用本实验目录下的 resume_<模型标签>.pt（原有逻辑）；
    #   填写路径 = 从该文件读取断点（**只读**：写入与结束时的删除仍作用于默认路径，
    #   该文件不会被覆盖或删除；须已存在，否则告警并回落默认）。
    #   环境变量 MEDDRA_RESUME_FILE 优先于本字段。
    resume_file: str = ""

@dataclass
class DataConfig:
    meddra_csv_path: str = "/kaggle/input/datasets/jiuqianqicc/meddra-29-0/MEDDRA_DATA_29_0.csv"
    cache_dir: str = "./meddra_cache"
    # 【新增】文本编码器（PubMedBERT/SapBERT/CODER）的本地目录，默认 ''（空）。
    #   非空：本地优先、离线加载（local_files_only=True，零联网）；若该目录尚无权重，
    #         则从 HF Hub 下载一次并 save_pretrained 落盘到该目录，之后每次运行都读本地；
    #   为空：保持原行为（按 HF Hub 名称加载，依赖 HuggingFace 自身缓存）。
    #   Kaggle 用法：把已下载好的模型目录做成 Dataset，挂载后将其路径填在此处即可。
    text_model_dir: str = ""
    force_rebuild_cache: bool = False
    force_rebuild_features: bool = False
    # 注：noGNN 无记忆库（Memory Bank）实现，故不提供 memory_k_per_soc /
    #     memory_sampling_strategy 配置（SAGE/GAT 版本才有，且默认关闭构建）。
    feature_type: str = "MedBERT"  # 'MedBERT' 或 'random'

# =============================================================================
# 1. GPU 设备管理器
# =============================================================================

class DeviceManager:
    def __init__(self):
        self.device = self._setup_device()
        self.num_gpus = 0
        self.gpu_names = []
        self._get_gpu_info()
        logger.info(f"使用设备: {self.device}")

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"CUDA 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA 不可用，使用 CPU")
        return device

    def _get_gpu_info(self) -> None:
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.num_gpus)]
            if self.num_gpus > 1:
                logger.info(f"检测到 {self.num_gpus} 块 GPU")
                for i, name in enumerate(self.gpu_names):
                    logger.info(f"  GPU {i}: {name}")

    def clear_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU 内存已清理")

device_manager = DeviceManager()
DEVICE = device_manager.device

# =============================================================================
# 2. 图缓存管理器（完整保留）
# =============================================================================

class GraphCacheManager:
    def __init__(self, cache_dir: str = "./meddra_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_info = {}
        self._load_cache_info()
        logger.info(f"缓存管理器初始化，目录: {cache_dir}")

    def _load_cache_info(self) -> None:
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
        info_path = os.path.join(self.cache_dir, "cache_info.json")
        try:
            # 【并发安全】先写临时文件再原子替换：多个进程共享同一 cache_dir 时直接写会互相截断
            # （后果只是"多重建一次图"，但能避免就避免；文件名带 PID 防两个进程互相踩临时文件）
            _tmp_info = f"{info_path}.tmp{os.getpid()}"
            with open(_tmp_info, "w", encoding="utf-8") as f:
                json.dump(self.cache_info, f, indent=2, ensure_ascii=False)
            try:
                os.replace(_tmp_info, info_path)
            finally:
                # 替换失败（Windows 下别的进程正打开目标文件时会失败；POSIX 不会）时不留下垃圾
                if os.path.exists(_tmp_info):
                    try:
                        os.remove(_tmp_info)
                    except OSError:
                        pass
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")

    def _get_file_hash(self, csv_path: str) -> Optional[str]:
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return None
        try:
            with open(csv_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:12]
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            return None

        
    # 缓存键中的版本 = 层级/标签构建版本（单一来源） + 本脚本后缀
    CACHE_VERSION = BUILD_HIERARCHY_VERSION + "-noGNN"
    def get_cache_key(self, csv_path: str, embedding_dim: int = 256) -> Optional[str]:
        file_hash = self._get_file_hash(csv_path)
        if file_hash is None:
            return None
        return f"meddra_{file_hash}_{self.CACHE_VERSION}_dim{embedding_dim}"

    def has_cache(self, csv_path: str, embedding_dim: int = 256) -> bool:
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return False
        graph_path = os.path.join(self.cache_dir, f"{key}_graph.pt")
        meta_path = os.path.join(self.cache_dir, f"{key}_metadata.pkl")
        if not (os.path.exists(graph_path) and os.path.exists(meta_path)):
            return False
        # ---- 【版本差异检测】----
        # 文件名里的版本号只能「隔离」，无法发现「改了层级/标签逻辑却忘记升号」。
        # 这里读取缓存元数据中记录的版本并比较：不一致即判定缓存失效（返回 False），
        # 由调用方走重建路径，避免静默复用旧标签／旧主父口径。
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


    def save_graph(self, csv_path: str, data: HeteroData, metadata: Dict,
                   embedding_dim: int = 256) -> bool:
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

    def save_train_history(self, csv_path: str, embedding_dim: int = 256,
                           train_history: Optional[dict] = None,
                           tag: str = "",
                           out_dir: str = "") -> bool:
        """保存训练历史。【v7.1-noGNN 修复】新增 tag 参数，避免实验互相覆盖。
        【每实验一个文件夹】out_dir 非空时写到该目录（本实验目录）。"""
        if train_history is None:
            return False
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return False
        # 将 numpy 类型转换为 Python 原生类型以便 JSON 序列化
        serializable_history = {}
        for k, v in train_history.items():
            if k == "hierarchical_sim_dict":
                serializable_history[k] = [
                    {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                     for kk, vv in d.items()}
                    for d in v
                ]
            elif isinstance(v, list) and v and isinstance(v[0], (np.floating, float, np.integer)):
                serializable_history[k] = [float(x) for x in v]
            else:
                serializable_history[k] = v
        # 【每实验一个文件夹】out_dir 非空 → 写到本实验目录；为空则退回缓存目录（兼容旧行为）
        target_dir = out_dir or self.cache_dir
        os.makedirs(target_dir, exist_ok=True)
        # 【v7.1-noGNN】加 tag 后缀
        suffix = f"_{tag}" if tag else ""
        file_path = os.path.join(target_dir, f"{key}_train_history{suffix}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            logger.info(f"训练历史已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")
            return False

    def load_graph(self, csv_path: str, embedding_dim: int = 256) -> Tuple[Optional[HeteroData], Optional[Dict]]:
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
            data = torch.load(graph_path, map_location="cpu", weights_only=False)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            # 【版本差异检测】防御性二次校验（防止绕过 has_cache 直接调用本函数）
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

    def save_pubmedbert_features(self, csv_path: str, features_dict: Dict[str, torch.Tensor],
                                 embedding_dim: int = 256,
                                 feature_model: str = "PubMedBERT") -> bool:
        """【v7.0-noGNN 修复】feature_model 加入缓存文件名，避免不同模型互相覆盖"""
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return False
        feat_path = os.path.join(self.cache_dir, f"{key}_{feature_model}_features.pt")
        try:
            save_data = {
                'features': features_dict,
                'csv_hash': self._get_file_hash(csv_path),
                'embedding_dim': embedding_dim,
                'feature_model': feature_model,
                'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'num_nodes': {k: v.shape[0] for k, v in features_dict.items()},
            }
            torch.save(save_data, feat_path)
            logger.info(f"✅ {feature_model} 特征已缓存: {feat_path}")
            return True
        except Exception as e:
            logger.error(f"保存 {feature_model} 特征失败: {e}")
            return False

    def load_pubmedbert_features(self, csv_path: str, embedding_dim: int = 256,
                                  feature_model: str = "PubMedBERT") -> Optional[Dict[str, torch.Tensor]]:
        """【v7.0-noGNN 修复】feature_model 加入缓存文件名"""
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return None
        feat_path = os.path.join(self.cache_dir, f"{key}_{feature_model}_features.pt")
        if not os.path.exists(feat_path):
            return None
        try:
            save_data = torch.load(feat_path, map_location="cpu", weights_only=False)
            current_hash = self._get_file_hash(csv_path)
            if save_data.get('csv_hash') != current_hash:
                logger.warning(f"{feature_model} 特征缓存与当前数据版本不匹配，将重新生成")
                return None
            if save_data.get('feature_model') != feature_model:
                logger.warning(f"缓存 feature_model={save_data.get('feature_model')} 与请求 {feature_model} 不匹配")
                return None
            logger.info(f"✅ {feature_model} 特征已从缓存加载: {feat_path}")
            return save_data['features']
        except Exception as e:
            logger.error(f"加载 {feature_model} 特征失败: {e}")
            return None

    def has_pubmedbert_cache(self, csv_path: str, embedding_dim: int = 256,
                             feature_model: str = "PubMedBERT") -> bool:
        """【v7.0-noGNN 修复】feature_model 加入缓存文件名"""
        key = self.get_cache_key(csv_path, embedding_dim)
        if key is None:
            return False
        feat_path = os.path.join(self.cache_dir, f"{key}_{feature_model}_features.pt")
        return os.path.exists(feat_path)

    # 【预留·未接线】缓存列表查询接口：当前三个训练流程均未调用（仅为后续增量训练预留）。
    def list_cache(self) -> None:
        if not self.cache_info:
            logger.info("暂无缓存")
            return
        logger.info("缓存列表:")
        for key, info in self.cache_info.items():
            logger.info(f"  {key} (保存时间: {info.get('save_time', '未知')})")

# =============================================================================
# 3. MedDRA 数据处理器（完整保留）
# =============================================================================

class HeteroMedDRADataProcessor:
    def __init__(self, meddra_csv_path: str, cache_manager: Optional[GraphCacheManager] = None):
        self.meddra_csv_path = meddra_csv_path
        self.cache_manager = cache_manager or GraphCacheManager()
        self.df = None
        self.hierarchy_map = {}
        self.name_map = {}
        self.node_soc_map = {}
        self.level_map = {"SOC": 0, "HLGT": 1, "HLT": 2, "PT": 3, "LLT": 4}
        self.level_names = ["SOC", "HLGT", "HLT", "PT", "LLT"]
        self.level_names_lower = [x.lower() for x in self.level_names]
        self.level_encoders = {}
        self.level_nodes = {}
        self.level_node_ids = {}
        self.num_nodes_by_level = {}
        self.num_all_nodes = 0
        self.node_codes = []
        self.node_levels = []
        self.node_names = []
        self.node_indices = {}
        self.edge_types = [
            ("soc", "has", "hlgt"),
            ("hlgt", "has", "hlt"),
            ("hlt", "has", "pt"),
            ("pt", "has", "llt"),
        ]
        self.graph_data = None
        self.build_time = None
        self._primary_parent = {}   # 主父映射（PRIMARY_SOC_FG='Y' 首次出现），供表兄弟硬负对照使用

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
        # ---- 尝试从缓存加载 ----
        if not force_rebuild and self.cache_manager.has_cache(self.meddra_csv_path, embedding_dim):
            if verbose:
                logger.info("发现缓存的图数据，正在加载...")
            data, metadata = self.cache_manager.load_graph(self.meddra_csv_path, embedding_dim)
            if data is not None:
                self.graph_data = data
                self._restore_metadata(metadata)   # 恢复层级/标签/主父等元数据
                self.build_time = metadata.get("build_time", "未知")
                return data, metadata

        # ---- 缓存不存在或强制重建：走 build_graph ----
        if verbose:
            logger.info("未找到缓存或强制重建，开始构建图...")
        return self.build_graph(embedding_dim, verbose)


    def build_graph(self, embedding_dim: int = 256,
                    verbose: bool = True) -> Tuple[HeteroData, Dict]:
        start_time = time.time()
        self.load_data()
        self.build_hierarchy()
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
            "primary_parent": self._primary_parent,        # 【新增】主父映射（与 SAGE/GAT 对齐）
            "build_hierarchy_version": BUILD_HIERARCHY_VERSION,  # 【版本差异检测】单一来源版本号
            "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_duration": time.time() - start_time,
        }
        self.cache_manager.save_graph(self.meddra_csv_path, data, metadata, embedding_dim)
        self.graph_data = data
        self.build_time = metadata["build_time"]
        if verbose:
            logger.info(f"图构建完成，耗时: {metadata['build_duration']:.2f} 秒")
        return data, metadata

    def _restore_metadata(self, metadata: Dict) -> None:
        for key in [
            "hierarchy_map", "name_map", "node_soc_map", "level_encoders",
            "level_nodes", "level_node_ids", "num_nodes_by_level",
            "num_all_nodes", "node_codes", "node_levels",
            "node_names", "node_indices",
        ]:
            setattr(self, key, metadata.get(key, {} if key != "num_all_nodes" else 0))
        # 【新增】恢复主父映射（与 SAGE/GAT 对齐）
        self._primary_parent = metadata.get("primary_parent", {})

    def load_data(self) -> pd.DataFrame:
        logger.info(f"加载 MedDRA 数据从: {self.meddra_csv_path}")
        if not os.path.exists(self.meddra_csv_path):
            raise FileNotFoundError(f"文件不存在: {self.meddra_csv_path}")
        self.df = pd.read_csv(self.meddra_csv_path)
        logger.info(f"数据形状: {self.df.shape}")

        # 【新增】验证 PRIMARY_SOC_FG 字段（与 SAGE 一致）
        if "PRIMARY_SOC_FG" not in self.df.columns:
            logger.warning("⚠️ PRIMARY_SOC_FG 字段不存在，将使用第一个 SOC 作为标签")
        else:
            unique_flags = self.df["PRIMARY_SOC_FG"].dropna().unique()
            logger.info(f"PRIMARY_SOC_FG 取值: {unique_flags}")
            n_primary = (self.df["PRIMARY_SOC_FG"] == "Y").sum()
            logger.info(f"主 SOC 行数: {n_primary} / {len(self.df)}")

        # 【新增】多父结构诊断（与 SAGE 一致）
        self.diagnose_multi_parent()

        return self.df


    def diagnose_multi_parent(self) -> Dict[str, Dict]:
        """多父节点统计诊断（用于论文数据报告）"""
        results = {}
        for level, col in [("HLGT", "HLGT_CODE"), ("HLT", "HLT_CODE"),
                           ("PT", "PT_CODE"), ("LLT", "LLT_CODE")]:
            node_parents = defaultdict(set)
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

            total = len(node_parents)
            multi_parent = sum(1 for p in node_parents.values() if len(p) > 1)
            dist = defaultdict(int)
            for parents in node_parents.values():
                dist[len(parents)] += 1
            results[level] = {
                "total": total,
                "multi_parent": multi_parent,
                "ratio": multi_parent / total if total > 0 else 0.0,
                "dist": dict(sorted(dist.items())),
            }

        logger.info("=" * 60)
        logger.info("MedDRA 多父节点统计 (DAG 结构)")
        logger.info("=" * 60)
        for level, stats in results.items():
            logger.info(f"  {level}: {stats['multi_parent']}/{stats['total']} "
                        f"({stats['ratio']*100:.2f}%) 多父节点")
            logger.info(f"       父节点数分布: {stats['dist']}")
        return results


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

        # 1. 为每个层级创建 LabelEncoder
        for level in self.level_names:
            col = f"{level}_CODE"
            if col in self.df.columns:
                codes = self.df[col].dropna().astype(int).unique().tolist()
                codes = sorted(codes)  # 与 LabelEncoder 的顺序对齐
                enc = LabelEncoder()
                enc.fit(codes)
                self.level_encoders[level] = enc
                self.level_nodes[level] = codes
                self.num_nodes_by_level[level] = len(codes)
                logger.info(f"  {level}: {len(codes)} 个节点")

        # 2. 遍历所有行，建立父子关系（不区分 Y/N，保留 DAG 完整性）
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
                pt_soc = self._safe_int(row, "PT_SOC_CODE")  # ★ 官方 PT 主 SOC

                # 【关键】用 PT_SOC_CODE 记录 PT 的官方 SOC（setdefault 保留首次，同一 PT 应唯一）
                if pt is not None and pt_soc is not None:
                    self.node_soc_map.setdefault(pt, pt_soc)

                # 主父记录（仅用于对比学习，不用来打 SOC 标签）
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

                # 父子关系（所有行，保留 DAG）
                self._add_relationship(llt, pt, "LLT")
                self._add_relationship(pt, hlt, "PT")
                self._add_relationship(hlt, hlgt, "HLT")
                self._add_relationship(hlgt, soc, "HLGT")

                # 名称
                self._store_names(row, llt, "LLT")
                self._store_names(row, pt, "PT")
                self._store_names(row, hlt, "HLT")
                self._store_names(row, hlgt, "HLGT")
                self._store_names(row, soc, "SOC")
            except Exception as e:
                logger.warning(f"跳过一行：{e}")
                continue

        logger.info(f"主 SOC 行数: {primary_count}, 非主 SOC 行数: {non_primary_count}")

        # 3. 整理全局节点列表
        nodes = sorted(list(self.hierarchy_map.keys()))
        self.node_codes = nodes
        self.node_levels = [self.hierarchy_map[n]["level"] for n in nodes]
        self.node_names = [self.name_map.get(n, str(n)) for n in nodes]
        self.node_indices = {n: i for i, n in enumerate(nodes)}
        self.num_all_nodes = len(nodes)

        for level in self.level_names:
            self.level_node_ids[level] = [
                self.node_indices[n] for n in nodes if self.hierarchy_map[n]["level"] == level
            ]
        logger.info(f"层级结构构建完成: 总节点数 {self.num_all_nodes}")

        # 4. PT 已从 PT_SOC_CODE 获得 SOC 标签
        pt_with_label = sum(1 for pt in self.level_nodes.get("PT", []) if pt in self.node_soc_map)
        logger.info(f"PT 官方 SOC 标签: {pt_with_label}/{len(self.level_nodes.get('PT', []))}")

        # 5. LLT 继承父 PT 的 SOC
        llt_inherited = 0
        for llt_code in self.level_nodes.get("LLT", []):
            parents = self.hierarchy_map.get(llt_code, {}).get("parent", [])
            if parents and parents[0] in self.node_soc_map:
                self.node_soc_map[llt_code] = self.node_soc_map[parents[0]]
                llt_inherited += 1
        logger.info(f"LLT SOC 标签（继承父 PT）: {llt_inherited}/{len(self.level_nodes.get('LLT', []))}")

        # 6. HLT 从子 PT 投票聚合
        for hlt_code in self.level_nodes.get("HLT", []):
            children_pt = self.hierarchy_map.get(hlt_code, {}).get("children", [])
            votes = [self.node_soc_map[c] for c in children_pt if c in self.node_soc_map]
            if votes:
                self.node_soc_map[hlt_code] = Counter(votes).most_common(1)[0][0]

        # 7. HLGT 从子 HLT 投票聚合
        for hlgt_code in self.level_nodes.get("HLGT", []):
            children_hlt = self.hierarchy_map.get(hlgt_code, {}).get("children", [])
            votes = [self.node_soc_map[c] for c in children_hlt if c in self.node_soc_map]
            if votes:
                self.node_soc_map[hlgt_code] = Counter(votes).most_common(1)[0][0]

        # 8. SOC 标签覆盖率
        non_soc_nodes = [n for n in self.node_codes
                        if self.hierarchy_map[n]["level"] != "SOC"]
        labeled_nodes = [n for n in non_soc_nodes if n in self.node_soc_map]
        logger.info(f"SOC 标签覆盖率: {len(labeled_nodes)}/{len(non_soc_nodes)} "
                    f"({len(labeled_nodes)/len(non_soc_nodes)*100:.2f}%)")

        # 9. SOC 分布诊断（关键）
        soc_counts = Counter(self.node_soc_map.values())
        total = sum(soc_counts.values())
        logger.info("=" * 60)
        logger.info("SOC 标签分布（Top 5）:")
        for soc, cnt in soc_counts.most_common(5):
            logger.info(f"  SOC {soc} ({self.get_node_name(soc)}): {cnt} ({cnt/total*100:.1f}%)")
        logger.info(f"最大 SOC 占比: {soc_counts.most_common(1)[0][1] / total * 100:.1f}%")
        logger.info("=" * 60)

        # 10. 主父覆盖率
        has_primary = [n for n in non_soc_nodes if n in self._primary_parent]
        logger.info(f"主父覆盖率: {len(has_primary)}/{len(non_soc_nodes)} "
                    f"({len(has_primary)/len(non_soc_nodes)*100:.2f}%)")

        return self.hierarchy_map, self.name_map

    # 【已废弃 / DEPRECATED】当前标签构建已改用 v7.2 分层策略（PT 用官方 PT_SOC_CODE、
    # LLT 继承父 PT、HLT/HLGT 由子节点投票聚合），本方法在 v7.2 之后的三个训练脚本中
    # 均不再被调用，仅为兼容旧版代码保留（其调用点见归档脚本 meddra/old/）。
    def _infer_missing_soc_labels(self) -> None:
        """
        [已废弃·无调用] 为未获得 SOC 标签的节点（HLT/HLGT）从父节点继承 SOC。
        MedDRA 29.0 有约 184 个节点（173 HLT + 11 HLGT）从未以
        PRIMARY_SOC_FG='Y' 出现，本方法按层级从高到低回退。
        """
        level_order = {"HLGT": 1, "HLT": 2, "PT": 3, "LLT": 4}

        unlabeled = [
            n for n in self.node_codes
            if self.hierarchy_map[n]["level"] != "SOC"
            and n not in self.node_soc_map
        ]
        if not unlabeled:
            return

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
        val = row.get(col)
        return int(val) if pd.notna(val) else None

    # 【已废弃 / DEPRECATED】v7.2 起 SOC 映射由 build_hierarchy 直接写入 self.node_soc_map，
    # 本方法在三个训练脚本中均不再被调用（调用点见归档脚本 meddra/old/）。
    def _set_node_soc_code(self, node_code, soc_code, overwrite=False):
        """
        [已废弃·无调用] 默认保留首次赋值，仅在 overwrite=True 时覆盖。
        与 SAGE 版本语义完全一致。
        """
        if node_code is None or soc_code is None:
            return
        if not overwrite and node_code in self.node_soc_map:
            return
        self.node_soc_map[node_code] = soc_code



    def _add_relationship(self, child: Optional[int], parent: Optional[int],
                          child_level: str) -> None:
        if child is None or parent is None:
            return
        for node in [child, parent]:
            if node not in self.hierarchy_map:
                if node == child:
                    level = child_level
                else:
                    level = self._get_parent_level(child_level)
                self.hierarchy_map[node] = {"parent": [], "children": [], "level": level}
        if parent not in self.hierarchy_map[child]["parent"]:
            self.hierarchy_map[child]["parent"].append(parent)
        if child not in self.hierarchy_map[parent]["children"]:
            self.hierarchy_map[parent]["children"].append(child)

    def _get_parent_level(self, child_level: str) -> str:
        order = ["LLT", "PT", "HLT", "HLGT", "SOC"]
        idx = order.index(child_level)
        return order[idx + 1] if idx + 1 < len(order) else "SOC"

    def _store_names(self, row: pd.Series, code: Optional[int], level: str) -> None:
        if code is None:
            return
        name = row.get(f"{level}_NAME_EN")
        if pd.isna(name):
            name = row.get(f"{level}_NAME")
        if pd.notna(name):
            self.name_map[code] = str(name)

    def create_hetero_graph(self, feature_dim: int = 128) -> HeteroData:
        logger.info("创建异构图...")
        data = HeteroData()
        for level in self.level_names:
            lvl = level.lower()
            nodes = self.level_nodes.get(level, [])
            if not nodes:
                continue
            data[lvl].x = torch.randn(len(nodes), feature_dim)
            data[lvl].node_codes = nodes
            logger.info(f"    {level}: {len(nodes)} 个节点")

        edge_pairs = [("SOC", "HLGT"), ("HLGT", "HLT"), ("HLT", "PT"), ("PT", "LLT")]
        for parent_level, child_level in edge_pairs:
            parent_lower, child_lower = parent_level.lower(), child_level.lower()
            parent_enc = self.level_encoders[parent_level]
            child_enc = self.level_encoders[child_level]
            edges = []
            parent_nodes = self.level_nodes.get(parent_level, [])
            child_set = set(self.level_nodes.get(child_level, []))
            for parent in parent_nodes:
                if parent not in self.hierarchy_map:
                    continue
                p_idx = parent_enc.transform([parent])[0]
                for child in self.hierarchy_map[parent].get("children", []):
                    if child in child_set:
                        c_idx = child_enc.transform([child])[0]
                        edges.append([p_idx, c_idx])
            if edges:
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[parent_lower, "has", child_lower].edge_index = edge_tensor
                logger.info(f"    {parent_level}→{child_level}: {len(edges)} 条边")

        for edge_type in list(data.edge_types):
            src, rel, dst = edge_type
            reverse_type = (dst, rel, src)
            if reverse_type not in data.edge_types and hasattr(data[edge_type], "edge_index"):
                data[reverse_type].edge_index = torch.stack(
                    [data[edge_type].edge_index[1], data[edge_type].edge_index[0]], dim=0
                )

        logger.info(f"异构图构建完成! 节点类型: {list(data.node_types)}")
        return data

    # 注：原 _build_query_caches 与 _parent_cache 已随 get_parent 一并移除——
    #     noGNN 无对比学习，从不消费父节点缓存；主父信息直接读 self._primary_parent。
    def get_soc_node_code(self, node_code: int) -> Optional[int]:
        if node_code is None:
            return None
        return self.node_soc_map.get(node_code, None)

    def get_node_level(self, node_code: int) -> str:
        return self.hierarchy_map.get(node_code, {}).get("level", "UNKNOWN")

    def get_node_name(self, node_code: int) -> str:
        return self.name_map.get(node_code, str(node_code))

    def get_nodes_by_level(self, level: str) -> List[int]:
        return self.level_nodes.get(level, [])

    def get_node_names_dict(self) -> Dict[str, List[str]]:
        return {
            l.lower(): [self.get_node_name(n) for n in self.level_nodes.get(l, [])]
            for l in self.level_names
        }

    def get_all_node_codes(self) -> Set[int]:
        return set(self.node_codes)

    # 【预留·未接线】新旧版本节点差集检测：当前训练流程未调用（仅为后续增量训练预留）。
    def get_new_nodes(self, old_processor: "HeteroMedDRADataProcessor") -> List[int]:
        old_codes = old_processor.get_all_node_codes()
        return [n for n in self.node_codes if n not in old_codes]

    # 【预留·未接线】新旧版本节点差集检测：当前训练流程未调用（仅为后续增量训练预留）。
    def get_removed_nodes(self, old_processor: "HeteroMedDRADataProcessor") -> List[int]:
        new_codes = self.get_all_node_codes()
        return [n for n in old_processor.node_codes if n not in new_codes]

# =============================================================================
# 4. 编码器（无 GNN，无对比学习）
# =============================================================================

class HeteroMedDRAEncoder(nn.Module):
    def __init__(
        self,
        node_types: List[str],
        num_socs: int,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.node_types = node_types
        self.embedding_dim = model_config.embedding_dim
        self.hidden_dim = model_config.hidden_dim
        self.num_socs = num_socs

        # 每个节点类型的特征投影层
        self.node_lin = nn.ModuleDict()
        for nt in node_types:
            self.node_lin[nt] = nn.Sequential(
                Linear(model_config.in_channels, model_config.hidden_dim),
                nn.LayerNorm(model_config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(model_config.dropout),
            )

        # 输出归一化
        self.output_norm = nn.LayerNorm(model_config.embedding_dim)
        # SOC 分类头
        self.soc_classifier = nn.Linear(model_config.embedding_dim, num_socs)
        self.dropout = nn.Dropout(model_config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播：仅投影 + 分类
        返回: (嵌入字典, 分类logits字典)
        """
        x_proj = {}
        for nt in self.node_types:
            if nt in x_dict:
                x_proj[nt] = self.node_lin[nt](x_dict[nt])

        x_out = {}
        for nt in self.node_types:
            if nt in x_proj:
                x_out[nt] = self.output_norm(x_proj[nt])

        soc_logits = {
            nt: self.soc_classifier(x_out[nt]) for nt in x_out if nt in self.node_types
        }
        return x_out, soc_logits

# =============================================================================
# 5. 训练器（仅分类损失，但验证时计算簇内相似度）
# =============================================================================

class HeteroMedDRATrainer:
    def __init__(
        self,
        model: HeteroMedDRAEncoder,
        data: HeteroData,
        processor: HeteroMedDRADataProcessor,
        training_config: TrainingConfig,
        inductive_holdout: Optional[Set[int]] = None,
    ):
        self.model = model.to(DEVICE)
        self.data = data.to(DEVICE)
        self.processor = processor
        self.training_config = training_config

        self.node_types = list(data.node_types)
        self.idx_to_level = {v: k for k, v in processor.level_map.items()}
        self.num_socs = len(processor.level_nodes.get("SOC", []))
        self.soc_list = processor.level_nodes.get("SOC", [])
        self.soc_to_idx = {soc: i for i, soc in enumerate(self.soc_list)}

        self.all_nodes = []
        self.all_levels = []
        self.all_node_types = []
        self.all_soc_labels = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []   # disjoint 协议下 = 留出集（只在选定权重上评估一次）
        self.num_train_nodes = 0
        # 【归纳式】留出集（节点 code 集合）；空集合表示转导式（本文件恒非空）
        self.inductive_holdout: Set[int] = set(inductive_holdout or ())
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
            # 注：noGNN 只有单一分类损失、无对比损失，故不设 train_ce_loss / train_cl_loss
            #     （原先这两个键只初始化、从不写入也从不读取，属死键，已移除）
            "epoch_times": [],
            "learning_rates": [],
            "intra_soc_sim": [],
            "hierarchical_sim_dict": [],
        }
        # 【新增】最佳模型对应的指标
        self.best_metrics = {}
        self.best_val_acc = 0.0
        self.best_macro_f1 = 0.0     # F1
        self.best_epoch = 0
        self.best_model_state = None
        # 【新增】早停 / 模型选择配置（可在训练前覆盖）
        self.es_primary_metric = "acc"        # 主判据："acc"(默认, 实测信噪比更高) 或 "macro_f1"
        self.es_min_epochs = 60               # 早停保护期：前 N 轮只记录、不累计 patience（≈计划轮数的30%）
        self.es_loss_min_delta = 1e-4         # 验证损失“仍在下降”的判定阈值
        self.best_val_loss = float("inf")     # 历史最优验证损失（用于“仍在学习”判定）
        self.es_min_delta = 0.001             # 主指标“严格提升”阈值（实测校准：0.002 会在平台期误判无进展）
        self.es_tolerance = 0.005             # 容差带：主指标不落后最优超过该值即视为“同水平”
        self.es_gap_eps = 1e-4                # 容差带内择优所需的最小 gap 提升
        self.collapse_weight = 3              # 塌缩轮等价计为几轮无进展（替代独立的塌缩停止阈值）
        self.collapse_eps = 1e-3              # intra 相对参考值上升的最小量（抑制噪声）
        self.collapse_gap_ratio = 0.1         # gap 相对参考值的最小下降比例（10%）
        self._collapse_ref_intra = None       # 参考 intra（随“选中最佳权重”更新）
        self._collapse_ref_gap = None         # 参考 gap（随“选中最佳权重”更新）
        self.best_primary = 0.0               # 主判据历史最优
        self.best_gap = 0.0                   # 选中模型的 gap_average

    def _prepare_training_data(self) -> None:
        """按 SOC 分层采样划分训练/验证集（使用 RandomState 保证可复现）"""
        logger.info("=" * 60)
        logger.info("准备训练数据（按 SOC 分层采样）")
        logger.info("=" * 60)

        for nt in self.node_types:
            if hasattr(self.data[nt], "node_codes"):
                for code in self.data[nt].node_codes:
                    level = self.processor.get_node_level(code)
                    if level and level in self.processor.level_map:
                        if level == "SOC":
                            continue
                        self.all_nodes.append(code)
                        self.all_levels.append(self.processor.level_map[level])
                        self.all_node_types.append(nt)
                        soc = self.processor.get_soc_node_code(code)
                        if soc is not None and soc in self.soc_to_idx:
                            self.all_soc_labels.append(self.soc_to_idx[soc])
                        else:
                            # 未匹配到 SOC 索引的节点（正常数据不应出现，仅作兜底）
                            self.all_soc_labels.append(-1)

        self.num_train_nodes = len(self.all_nodes)
        logger.info(f"总节点数: {self.num_train_nodes:,}")
        for nt in self.node_types:
            cnt = sum(1 for t in self.all_node_types if t == nt)
            logger.info(f"    {nt}: {cnt:,}")

        soc_to_indices = defaultdict(list)
        for idx, soc_lab in enumerate(self.all_soc_labels):
            if soc_lab >= 0:
                soc_to_indices[soc_lab].append(idx)

        train_idx, val_idx = [], []
        # 【关键修改】使用独立的 RandomState，与 SAGE 版划分方式完全一致
        rng = np.random.RandomState(self.training_config._split_seed)   # 划分种子（未设时 = random_state）

        for lab, indices in soc_to_indices.items():
            n_val = max(1, min(int(len(indices) * self.training_config.val_ratio),
                               len(indices) // 2))
            if len(indices) <= 2:
                val_sel = indices[:1]
                train_sel = indices[1:]
            else:
                val_sel = rng.choice(indices, n_val, replace=False)
                train_sel = [i for i in indices if i not in val_sel]
            val_idx.extend(val_sel)
            train_idx.extend(train_sel)

        # ---- 【归纳式】覆盖上面的分层抽样：留出子树 → 验证集，其余 → 训练集 ----
        test_idx = []   # 第一轮（holdout 协议）不设独立留出评估
        if INDUCTIVE_MODE:
            _ho = self.inductive_holdout
            if not _ho:
                raise RuntimeError("归纳式模式：留出集为空（检查 select_inductive_holdout 的调用）")
            if INDUCTIVE_SELECT_DISJOINT:
                # 【2026-10-29 disjoint】模型选择集取自非留出节点；留出集只在选定权重上评估一次
                train_idx, val_idx, test_idx = _inductive_disjoint_split(
                    self.all_nodes, self.all_soc_labels, _ho,
                    self.training_config.val_ratio, self.training_config._split_seed)
                if not train_idx or not val_idx or not test_idx:
                    raise RuntimeError(
                        f"归纳式 disjoint：划分异常（训练 {len(train_idx)} / 选择 {len(val_idx)} / 留出 {len(test_idx)}）"
                    )
                if set(val_idx) & set(test_idx) or set(train_idx) & set(test_idx):
                    raise RuntimeError("归纳式 disjoint：模型选择集或训练集与留出集重叠")
                logger.info(
                    f"【归纳式·disjoint】留出 {len(test_idx):,} 个节点实例（只在选定权重上评估一次）；"
                    f"模型选择集 {len(val_idx):,}（非留出节点按 SOC 分层，split_seed="
                    f"{self.training_config._split_seed}）；训练集 {len(train_idx):,}"
                )
            else:
                train_idx = [i for i, c in enumerate(self.all_nodes) if c not in _ho]
                val_idx = [i for i, c in enumerate(self.all_nodes) if c in _ho]
                if not train_idx or not val_idx:
                    raise RuntimeError(
                        f"归纳式模式：划分异常（训练 {len(train_idx)} / 验证 {len(val_idx)}）"
                    )
                logger.info(
                    f"【归纳式】留出整棵 HLT 子树 → 验证集 {len(val_idx):,} 个留出节点 / "
                    f"训练集 {len(train_idx):,} 个节点（不再按 SOC 分层抽样）"
                )

        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = list(test_idx)
        logger.info(f"训练集: {len(train_idx):,} 个节点")
        logger.info(f"验证集: {len(val_idx):,} 个节点")
        logger.info(f"验证集比例: {len(val_idx) / self.num_train_nodes * 100:.1f}%")

        # 各层训练/验证样本量（分层指标的样本量依据；与 evaluate() 输出的 {level}_n 同口径）
        # 说明：分层变量是 SOC 标签而非层级，各层占比是“同一 SOC 池内四层混合后随机抽样”的自然结果，
        #       故此处显式打印，便于解读分层指标、填写论文表格。SOC 层本就不进入训练/验证集。
        _tr_by_nt, _va_by_nt = defaultdict(int), defaultdict(int)
        for _i in train_idx:
            _tr_by_nt[self.all_node_types[_i]] += 1
        for _i in val_idx:
            _va_by_nt[self.all_node_types[_i]] += 1
        for _nt in self.node_types:
            _n_lv = _tr_by_nt[_nt] + _va_by_nt[_nt]
            if _n_lv == 0:
                continue
            logger.info(
                f"    {_nt}: 训练 {_tr_by_nt[_nt]:,} / 验证 {_va_by_nt[_nt]:,}"
                f"（验证占比 {_va_by_nt[_nt] / _n_lv * 100:.1f}%）"
            )



    def _get_node_index(self, node_code: int, node_type: str) -> Optional[int]:
        """获取节点在其层级中的局部索引"""
        try:
            level = node_type.upper()
            idx = self.processor.level_encoders[level].transform([node_code])[0]
            return int(idx)
        except Exception:
            return None

    # ============= 新增：层级簇内余弦相似度计算 =============
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
            3. 缺少主父信息时 cousin 返回 0.0（NoGNN 无对比学习，不依赖主父）。

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
            valid = [valid[i] for i in sel]

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
        i = rng.randint(0, M, n_pairs)
        j = rng.randint(0, M, n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return 0.0, 0.0

        cos = np.einsum("ij,ij->i", E[i], E[j])

        # inter：两个节点无任何共同父节点
        no_shared = np.fromiter(
            (len(parent_sets[a] & parent_sets[b]) == 0 for a, b in zip(i, j)),
            dtype=bool, count=len(i),
        )
        # cousin：共享祖父（非空）且主父不同
        cousin = np.fromiter(
            (
                grandparent[a] is not None
                and grandparent[a] == grandparent[b]
                and primary_parent[a] != primary_parent[b]
                for a, b in zip(i, j)
            ),
            dtype=bool, count=len(i),
        )

        inter = float(cos[no_shared].mean()) if no_shared.any() else 0.0
        cousin_sim = float(cos[cousin].mean()) if cousin.any() else 0.0
        return inter, cousin_sim

    def compute_hierarchical_intra_similarity(
        self, 
        embeddings_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        计算所有层级的簇内平均余弦相似度。
        优化点：
            1. 使用有效节点数（去重）进行加权平均
            2. 支持多父节点（如 PT 有多个 HLT）
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
        node_index_cache: Dict[str, Dict[int, int]] = {}
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            ntype = level.lower()
            nodes = self.processor.get_nodes_by_level(level)
            index_cache = {}
            for node in nodes:
                idx = self._get_node_index(node, ntype)
                if idx is not None:
                    index_cache[node] = idx
            node_index_cache[ntype] = index_cache

        emb_cache: Dict[str, Dict[int, np.ndarray]] = {}
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            ntype = level.lower()
            if ntype not in embeddings_dict:
                emb_cache[ntype] = {}
                continue
            emb_tensor = embeddings_dict[ntype]
            index_cache = node_index_cache.get(ntype, {})
            emb_cache[ntype] = {
                node: emb_tensor[idx].cpu().numpy()
                for node, idx in index_cache.items()
            }
        
        # ---- 2. 定义层级配置：当前层级 -> 分组依据的父层级 ----
        level_configs = [
            ('HLGT', 'SOC'),   # HLGT: 按 SOC 分组
            ('HLT', 'HLGT'),   # HLT: 按 HLGT 分组
            ('PT', 'HLT'),     # PT: 按 HLT 分组（多父节点支持）
            ('LLT', 'PT'),     # LLT: 按 PT 分组
        ]
        
        effective_counts = {}
        
        for current_level, target_level in level_configs:
            ntype = current_level.lower()
            nodes = self.processor.get_nodes_by_level(current_level)
            if len(nodes) < 2:
                results[current_level] = 0.0
                effective_counts[current_level] = 0
                continue
            
            group_to_embs = defaultdict(list)
            emb_dict = emb_cache.get(ntype, {})
            unique_nodes = set()
            
            for node in nodes:
                emb = emb_dict.get(node)
                if emb is None:
                    continue
                # 获取所有父节点（支持多父）
                parent_nodes = self.processor.hierarchy_map.get(node, {}).get("parent", [])
                if parent_nodes:
                    unique_nodes.add(node)
                for parent_node in parent_nodes:
                    if parent_node is not None:
                        group_to_embs[parent_node].append(emb)
            
            effective_counts[current_level] = len(unique_nodes)
            sims = []
            for embs in group_to_embs.values():
                if len(embs) < 2:
                    continue
                embs = np.vstack(embs)
                sim_mat = cosine_similarity(embs)
                triu = np.triu_indices_from(sim_mat, k=1)
                if triu[0].size > 0:
                    sims.append(np.mean(sim_mat[triu]))
            results[current_level] = np.mean(sims) if sims else 0.0
        
        # ---- 3. 计算加权平均（按有效节点数） ----
        total_nodes = 0
        weighted_sum = 0.0
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            n_nodes = effective_counts.get(level, self.processor.num_nodes_by_level.get(level, 0))
            total_nodes += n_nodes
            weighted_sum += results.get(level, 0) * n_nodes
        results['average'] = weighted_sum / total_nodes if total_nodes > 0 else 0.0

        # 均匀平均（可选）
        uniform_sum = sum(results.get(level, 0) for level in ['HLGT', 'HLT', 'PT', 'LLT'])
        results['uniform_average'] = uniform_sum / 4.0

        # ---- 5. 采样估计 inter（跨簇）与 cousin（表兄弟硬负）相似度，并计算 gap ----
        # gap = intra - inter：越大说明“组内近、组间远”，判别性越强；
        # 若发生表示塌缩，intra 与 inter 会同时升高，gap 反而缩小，可作为塌缩预警。
        inter_res: Dict[str, float] = {}
        cousin_res: Dict[str, float] = {}
        for level in ['HLGT', 'HLT', 'PT', 'LLT']:
            ntype = level.lower()
            inter_val, cousin_val = self._sampled_inter_and_cousin_similarity(
                self.processor.get_nodes_by_level(level),
                emb_cache.get(ntype, {}),
            )
            inter_res[level] = inter_val
            cousin_res[level] = cousin_val
            intra_val = float(results.get(level, 0.0))
            results[f"inter_{level}"] = inter_val
            results[f"cousin_{level}"] = cousin_val
            results[f"gap_{level}"] = intra_val - inter_val
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

        _inter_uniform = float(np.mean([inter_res[_lv] for _lv in ['HLGT', 'HLT', 'PT', 'LLT']]))
        results['inter_uniform_average'] = _inter_uniform
        results['gap_uniform_average'] = float(results.get('uniform_average', 0.0)) - _inter_uniform

        results['effective_counts'] = effective_counts
        
        return results

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

        说明：本基线不含图消息传递，理论上不存在经父链传播标签的通道，因此该诊断对
        NoGNN 应表现为两组准确率接近（Gap ≈ 0）。把它同样跑一遍可作为对照——若 SAGE/GAT
        的 Gap 显著大于 NoGNN，即为结构线索解释力的直接证据。该指标仅用于分析，
        不参与早停与模型选择。
        """
        if ntype not in self._DIAG_LEVELS:
            return None
        pp = self.processor._primary_parent.get(node_code, None)
        if pp is None:
            return None
        return pp in self._get_train_code_set()

    # ============= 修改 evaluate 方法，返回相似度字典 =============
    def evaluate(self, indices: Optional[List[int]] = None) -> Tuple[float, float, float, Dict, Dict]:
        """
        在验证集上评估模型。
        
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
        
        _eval_indices = self.val_indices if indices is None else indices   # 默认验证集；传留出集即最终评估
        self.model.eval()
        if len(_eval_indices) == 0:
            return 0.0, 0.0, 0.0, {}, {}
        
        x_dict = {nt: self.data[nt].x for nt in self.node_types
                  if hasattr(self.data[nt], "x")}
        with torch.no_grad():
            x_out, soc_logits = self.model(x_dict)
        
        total_loss = 0.0
        correct, total = 0, 0
        num_batches = 0
        batch_size = self.training_config.batch_size
        
        # 收集所有预测和真实标签
        all_preds, all_truths = [], []
        per_level_preds = defaultdict(list)
        per_level_truths = defaultdict(list)

        # 【诊断·方案①】结构线索通道计数器：按“主父是否在训练集内”分组
        diag_correct = {"in": 0, "out": 0}
        diag_total = {"in": 0, "out": 0}
        diag_correct_lv = {lv: {"in": 0, "out": 0} for lv in self._DIAG_LEVELS}
        diag_total_lv = {lv: {"in": 0, "out": 0} for lv in self._DIAG_LEVELS}
        
        for i in range(0, len(_eval_indices), batch_size):
            batch_idx = _eval_indices[i:i + batch_size]
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
                idxs = [self._get_node_index(n, ntype) for n in nodes]
                valid_idxs = [i for i in idxs if i is not None]
                if not valid_idxs:
                    continue
                valid_logits = logits[valid_idxs]
                valid_socs = torch.tensor(
                    [soc_by_type[ntype][j] for j, idx in enumerate(idxs) if idx is not None],
                    dtype=torch.long, device=DEVICE,
                )
                if len(valid_socs) > 0:
                    loss = F.cross_entropy(valid_logits, valid_socs)
                    total_loss += loss.item()
                    num_batches += 1
                    pred = valid_logits.argmax(dim=1)
                    correct += (pred == valid_socs).sum().item()
                    total += len(valid_socs)
                    
                    pred_list = pred.cpu().tolist()
                    truth_list = valid_socs.cpu().tolist()
                    all_preds.extend(pred_list)
                    all_truths.extend(truth_list)
                    per_level_preds[ntype].extend(pred_list)
                    per_level_truths[ntype].extend(truth_list)

                    # 【诊断·方案①】按节点粒度累计两个子群的正确数/总数
                    _valid_nodes = [n for n, idx in zip(nodes, idxs) if idx is not None]
                    _correct_list = (pred == valid_socs).cpu().tolist()
                    for _node, _ok in zip(_valid_nodes, _correct_list):
                        _flag = self._parent_supervised(_node, ntype)
                        if _flag is None:
                            continue
                        _key = "in" if _flag else "out"
                        diag_total[_key] += 1
                        diag_correct[_key] += int(_ok)
                        if ntype in diag_total_lv:
                            diag_total_lv[ntype][_key] += 1
                            diag_correct_lv[ntype][_key] += int(_ok)
        
        val_acc = correct / max(1, total)
        val_loss = total_loss / max(1, num_batches)
        
        metrics = {
            "accuracy": val_acc,
            "macro_f1": f1_score(all_truths, all_preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(all_truths, all_preds, average="weighted", zero_division=0),
            "macro_precision": precision_score(all_truths, all_preds, average="macro", zero_division=0),
            "macro_recall": recall_score(all_truths, all_preds, average="macro", zero_division=0),
        }
        
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
                    metrics[f"{ntype}_macro_f1"] = 0.0
                metrics[f"{ntype}_n"] = len(preds)
            else:
                metrics[f"{ntype}_acc"] = 0.0
                metrics[f"{ntype}_macro_f1"] = 0.0
                metrics[f"{ntype}_n"] = 0
        
        # 【诊断·方案①】结构线索通道：验证集按“主父是否在训练集内”分组的准确率对照。
        # 读法：NoGNN 无图消息传递，两组应接近（Gap≈0）；若 SAGE/GAT 的 Gap 明显更大，
        #       即为“结构线索（父链标签传播）”解释力的直接证据。适用层级 HLT/PT/LLT。
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

        metrics["confusion_matrix"] = confusion_matrix(all_truths, all_preds).tolist()
        
        hierarchical_sim_dict = self.compute_hierarchical_intra_similarity(x_out)
        intra_sim_avg = hierarchical_sim_dict.get('average', 0.0)
        
        return val_loss, val_acc, intra_sim_avg, hierarchical_sim_dict, metrics

    
    # ============= 训练主循环（修改，记录相似度） =============

    def save_holdout_predictions(self, tag: str) -> None:
        """【2026-10-29 disjoint】留出集逐节点对错 → holdout_predictions_<tag>.json（离线规则对照用）。"""
        if not self.test_indices or not self.soc_list:
            return
        self.model.eval()
        x_dict = {nt: self.data[nt].x for nt in self.node_types if hasattr(self.data[nt], "x")}
        with torch.no_grad():
            _, soc_logits = self.model(x_dict)
        rows = []
        for idx in self.test_indices:
            node = self.all_nodes[idx]
            ntype = self.all_node_types[idx]
            truth = self.all_soc_labels[idx]
            if truth < 0 or ntype not in soc_logits:
                continue
            local = self._get_node_index(node, ntype)
            if local is None:
                continue
            pred = int(soc_logits[ntype][local].argmax(dim=0).item())
            rows.append({
                "node_code": int(node),
                "level": ntype,
                "true_soc_code": self.soc_list[truth] if truth < len(self.soc_list) else None,
                "pred_soc_code": self.soc_list[pred] if pred < len(self.soc_list) else None,
                "correct": bool(pred == truth),
            })
        with open(f"holdout_predictions_{tag}.json", "w", encoding="utf-8") as _f:
            json.dump(rows, _f, ensure_ascii=False, indent=1)
        logger.info("📄 留出集逐节点预测已保存: holdout_predictions_%s.json（%d 条）", tag, len(rows))

    def train(self, verbose: bool = True, train_epoch_log: int = 1) -> Tuple[HeteroMedDRAEncoder, Dict]:
        cfg = self.training_config
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )

        if verbose:
            logger.info("=" * 60)
            logger.info("🚀 开始训练 (仅 SOC 分类，验证时计算簇内相似度) [noGNN]")
            logger.info(f"训练集: {len(self.train_indices):,} 个节点")
            logger.info(f"验证集: {len(self.val_indices):,} 个节点")
            logger.info(f"SOC 类别数: {self.num_socs}")
            logger.info("=" * 60)

        # ✅ 模型选择：主判据为 self.es_primary_metric（默认 "acc" = 验证准确率），
        #    容差 self.es_tolerance 内优先选 gap_average 更大的模型；并启用塌缩护栏。
        self.best_val_acc = 0.0         # 历史最高准确率（仅记录/报告）
        self.best_primary = 0.0         # 主判据历史最优
        self.best_gap = 0.0             # 选中模型的 gap_average
        self.best_epoch = 0
        self.best_macro_f1 = 0.0        # 选中模型的 macro_f1
        patience_counter = 0            # 无严格提升的连续轮数
        self.best_val_loss = float("inf")   # 历史最优验证损失（用于“仍在学习”判定）
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

            if epoch < cfg.warmup_epochs:
                warmup_lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
            else:
                scheduler.step()

            train_loss, train_acc = self.train_epoch(optimizer, epoch=epoch)

            if np.isnan(train_loss):
                logger.error(f"Epoch {epoch+1} 训练损失为 NaN，停止训练")
                break

            # 【修改】接收 5 个返回值
            val_loss, val_acc, intra_sim, hierarchical_sim_dict, metrics = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times"].append(time.time() - epoch_start)
            self.history["learning_rates"].append(optimizer.param_groups[0]["lr"])
            self.history["hierarchical_sim_dict"].append(hierarchical_sim_dict)
            self.history["intra_soc_sim"].append(intra_sim)

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

            if verbose and ((epoch + 1) % train_epoch_log == 0
                            or epoch == 0 or epoch == cfg.epochs - 1):
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} "
                            f"({self.history['epoch_times'][-1]:.2f}s)")
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
                logger.info(
                    f"  Hierarchical Sim: HLGT={hierarchical_sim_dict.get('HLGT',0):.4f}, "
                    f"HLT={hierarchical_sim_dict.get('HLT',0):.4f}, "
                    f"PT={hierarchical_sim_dict.get('PT',0):.4f}, "
                    f"LLT={hierarchical_sim_dict.get('LLT',0):.4f}, "
                    f"Avg={hierarchical_sim_dict.get('average',0):.4f}"
                )
                # 【新增】inter（跨簇）/gap（intra-inter）/表兄弟硬负 对照
                logger.info(
                    f"  Inter(跨簇采样): HLGT={hierarchical_sim_dict.get('inter_HLGT',0):.4f}, "
                    f"HLT={hierarchical_sim_dict.get('inter_HLT',0):.4f}, "
                    f"PT={hierarchical_sim_dict.get('inter_PT',0):.4f}, "
                    f"LLT={hierarchical_sim_dict.get('inter_LLT',0):.4f}, "
                    f"Avg={hierarchical_sim_dict.get('inter_average',0):.4f}"
                )
                logger.info(
                    f"  Gap(intra-inter): HLGT={hierarchical_sim_dict.get('gap_HLGT',0):.4f}, "
                    f"HLT={hierarchical_sim_dict.get('gap_HLT',0):.4f}, "
                    f"PT={hierarchical_sim_dict.get('gap_PT',0):.4f}, "
                    f"LLT={hierarchical_sim_dict.get('gap_LLT',0):.4f}, "
                    f"Avg={hierarchical_sim_dict.get('gap_average',0):.4f}"
                )
                logger.info(
                    f"  Cousin(表兄弟硬负): HLGT={hierarchical_sim_dict.get('cousin_HLGT',0):.4f}, "
                    f"HLT={hierarchical_sim_dict.get('cousin_HLT',0):.4f}, "
                    f"PT={hierarchical_sim_dict.get('cousin_PT',0):.4f}, "
                    f"LLT={hierarchical_sim_dict.get('cousin_LLT',0):.4f}, "
                    f"Avg={hierarchical_sim_dict.get('cousin_average',0):.4f}, "
                    f"GapCousin={hierarchical_sim_dict.get('gap_cousin_average',0):.4f}"
                )
                logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}, "
                            f"Patience: {patience_counter}/{cfg.patience}")

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

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(DEVICE)

        # ---- 【2026-10-29 disjoint】留出集只在选定权重上评估一次（不参与早停/模型选择）----
        self.holdout_metrics = {}
        if self.test_indices:
            _h_loss, _h_acc, _h_intra, _h_sim, _h_metrics = self.evaluate(self.test_indices)
            if _h_metrics:
                self.holdout_metrics = {k: v for k, v in _h_metrics.items() if k != "confusion_matrix"}
                self.holdout_metrics.update({
                    "holdout_accuracy": _h_acc, "holdout_loss": _h_loss,
                    "selected_epoch": int(self.best_epoch + 1),
                    "num_holdout_instances": int(len(self.test_indices)),
                    "num_select_instances": int(len(self.val_indices)),
                    "num_train_instances": int(len(self.train_indices)),
                    "selection_protocol": "disjoint",
                    "train_seed": int(self.training_config._train_seed),
                    "split_seed": int(self.training_config._split_seed),
                    "keep_edges": bool(INDUCTIVE_KEEP_EDGES),
                    "select_set_accuracy_at_selected_epoch": float(self.best_metrics.get("accuracy", 0.0)),
                })
                _h_tag = str(globals().get("_MODEL_TAG", "model"))
                try:
                    with open(f"holdout_metrics_{_h_tag}.json", "w", encoding="utf-8") as _hf:
                        json.dump(self.holdout_metrics, _hf, ensure_ascii=False, indent=2, default=float)
                    logger.info(f"📄 留出集指标已保存: holdout_metrics_{_h_tag}.json")
                except Exception as _he:
                    logger.warning(f"⚠️ 留出集指标写 JSON 失败: {_he}")
                try:
                    self.save_holdout_predictions(_h_tag)
                except Exception as _hpe:
                    logger.warning(f"⚠️ 留出集逐节点预测导出失败: {_hpe}")
                logger.info(
                    f"\n🧪 留出集评估（{len(self.test_indices):,} 个节点实例，选定权重 Epoch {self.best_epoch + 1}）："
                    f"Acc={_h_metrics.get('accuracy', 0):.4f}, Macro-F1={_h_metrics.get('macro_f1', 0):.4f}"
                )

        total_time = time.time() - start_time
        if verbose:
            logger.info(f"\n✅ 训练完成! 总耗时: {total_time / 60:.2f} 分钟")
            logger.info(
                f"选中权重: Epoch {self.best_epoch + 1}"
                f"（Acc={self.best_metrics.get('accuracy', 0):.4f}）；"
                f"历史最高验证准确率: {self.best_val_acc:.4f}"
            )
            logger.info(f"最佳 Macro-F1: {self.best_macro_f1:.4f}")
            if self.best_metrics:
                logger.info(
                    f"选中模型[{self.best_metrics.get('es_selected_by', 'strict')}] "
                    f"(主判据={self.best_metrics.get('es_primary', 'acc')}): "
                    f"Acc={self.best_metrics.get('accuracy', 0):.4f}, "
                    f"Macro-F1={self.best_metrics.get('macro_f1', 0):.4f}, "
                    f"Gap={self.best_metrics.get('gap_average', 0):.4f}"
                )

        return self.model, self.history

    
    # --------------------- 断点续训 ---------------------
    # 设计要点：
    #   1) 落盘"再跑一轮所需的一切"：模型 / 优化器 / 调度器 / 最佳权重 / 早停计数 / history / RNG；
    #   2) 指纹一致性：配置、协议参数、构建版本、图规模任一变化 → 拒绝续训（防止一次实验混两套设置）；
    #   3) 数值可复现性：训练本身按 `random_state*1000 + epoch` 逐轮播种（见 train_epoch），
    #      不依赖全局 RNG 态，故续训后的轨迹与"不中断训练"逐位一致；RNG 态仍一并保存，只为让训练
    #      结束后的绘图采样也可复现；
    #   4) 落盘用"临时名 + os.replace"原子替换，避免写到一半被杀导致断点文件损坏；
    #   5) 只含原生类型与张量（numpy 标量经 _resume_sanitize 转换），兼容 torch>=2.6 的
    #      torch.load(weights_only=True) 默认行为。
    #   6) noGNN 无对比学习 → 指纹中不含 contrastive_config。

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
            "num_train_nodes": int(self.num_train_nodes),
            "num_soc_classes": int(self.num_socs),
            # 【归纳式】留出集参与指纹：换了留出集就不允许复用断点（避免一次实验混两套划分）
            "inductive_holdout_digest": (len(self.inductive_holdout),
                                         sum(sorted(self.inductive_holdout)) % (10 ** 9)),
            # 【v1/v2 协议】是否保留留出节点的层级边 —— 协议不同即视为不同实验，禁止互相续训
            "inductive_keep_edges": bool(INDUCTIVE_KEEP_EDGES),
            "inductive_select_disjoint": bool(INDUCTIVE_SELECT_DISJOINT),
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


    def train_epoch(self, optimizer: torch.optim.Optimizer,
                    epoch: int = 0) -> Tuple[float, float]:
        """
        单 epoch 训练（仅分类损失）。

        【v7.1-noGNN 训练动态修复】
            原先每个 batch 都做一次全模型前向 + 一次 backward + 一次 step
            （mini-batch SGD，每 epoch 更新 ~211 次）。
            现改为：每个 epoch 只做一次全模型前向，累积所有 batch 的 loss
            后一次 backward + step（full-batch GD，每 epoch 更新 1 次）。
            与 SAGE/GAT v7.1 训练动态严格对齐，避免"训练动态"成为混淆变量。
        """
        self.model.train()
        cfg = self.training_config
        total_loss = 0.0
        correct, total = 0, 0
        num_batches = 0

        if len(self.train_indices) == 0:
            return 0.0, 0.0

        rng = np.random.RandomState(self.training_config._train_seed * 1000 + epoch)   # 训练种子（未设时 = random_state）
        indices = rng.permutation(self.train_indices)

        x_dict = {nt: self.data[nt].x for nt in self.node_types
                  if hasattr(self.data[nt], "x")}

        # ★ 性能关键：每个 epoch 只做一次全模型前向
        _, soc_logits = self.model(x_dict)
        batch_losses = []

        for i in range(0, len(indices), cfg.batch_size):
            batch_idx = indices[i:i + cfg.batch_size]
            batch_nodes = [self.all_nodes[idx] for idx in batch_idx]
            batch_socs = [self.all_soc_labels[idx] for idx in batch_idx]
            batch_types = [self.all_node_types[idx] for idx in batch_idx]

            by_type = defaultdict(list)
            soc_by_type = defaultdict(list)
            for n, s, t in zip(batch_nodes, batch_socs, batch_types):
                by_type[t].append(n)
                soc_by_type[t].append(s)

            ce_loss = 0.0
            ntypes_used = 0
            batch_correct, batch_total = 0, 0

            for t, nodes in by_type.items():
                if t not in soc_logits:
                    continue
                logits = soc_logits[t]
                idxs = [self._get_node_index(n, t) for n in nodes]
                valid_idxs = [i for i in idxs if i is not None]
                if not valid_idxs:
                    continue
                valid_logits = logits[valid_idxs]
                valid_socs = torch.tensor(
                    [soc_by_type[t][j] for j, idx in enumerate(idxs) if idx is not None],
                    dtype=torch.long, device=DEVICE,
                )
                if len(valid_socs) > 0:
                    loss_ce = F.cross_entropy(valid_logits, valid_socs)
                    ce_loss += loss_ce
                    ntypes_used += 1
                    pred = valid_logits.argmax(dim=1)
                    batch_correct += (pred == valid_socs).sum().item()
                    batch_total += len(valid_socs)

            if batch_total > 0 and ntypes_used > 0:
                ce_loss = ce_loss / ntypes_used
            else:
                ce_loss = torch.tensor(0.0, device=DEVICE)

            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                logger.warning("检测到 NaN/Inf 损失，跳过此批次")
                continue

            batch_losses.append(ce_loss)
            total_loss += ce_loss.item()
            correct += batch_correct
            total += batch_total
            num_batches += 1

        # ★ 所有 batch 处理完后，一次性 backward + step
        if batch_losses:
            optimizer.zero_grad()
            total_batch_loss = torch.stack(batch_losses).mean()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            optimizer.step()

        if num_batches == 0:
            return 0.0, 0.0

        avg_loss = total_loss / num_batches
        avg_acc = correct / max(1, total)
        return avg_loss, avg_acc

    def generate_embeddings(self) -> Dict[str, pd.DataFrame]:
        """生成所有节点的最终嵌入"""
        self.model.eval()
        x_dict = {nt: self.data[nt].x for nt in self.node_types if hasattr(self.data[nt], "x")}
        with torch.no_grad():
            x_out, _ = self.model(x_dict)

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
                result[nt.upper()] = df
                all_embs.append(embs)
                all_codes.extend(codes)
                all_levels.extend([self.processor.get_node_level(c) for c in codes])

        if all_embs:
            all_embs = np.vstack(all_embs)
            full_df = pd.DataFrame(
                all_embs,
                index=all_codes,
                columns=[f"dim_{i}" for i in range(self.model.embedding_dim)],
            )
            full_df["level"] = all_levels
            full_df["node_name"] = [self.processor.get_node_name(c) for c in all_codes]
            result["all"] = full_df

        logger.info(f"生成了 {len(result)} 个层级的嵌入")
        return result

# =============================================================================
# 6. 可视化工具（包含相似度曲线绘制）
# =============================================================================

class Visualizer:
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], 
                              feature_model: str = "PubMedBERT",
                              save_path: str = "training_history_noGNN.png") -> None:
        """
        绘制训练历史曲线，包括损失、准确率和层级相似度。
        """
        if not history.get("train_loss"):
            logger.warning("无训练历史数据，无法绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"MedDRA Training History ({feature_model})", fontsize=16, fontweight="bold")

        # ---- 1. 损失曲线 ----
        ax1 = axes[0, 0]
        ax1.plot(history["train_loss"], label="Train Loss", linewidth=2, color="blue")
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=2, color="red", linestyle="--")
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

        # ---- 3. 层级簇内相似度曲线（新增） ----
        ax3 = axes[1, 0]
        if history.get("hierarchical_sim_dict"):
            # 提取各层级相似度
            for level in ['HLGT', 'HLT', 'PT', 'LLT']:
                values = [d.get(level, 0) for d in history["hierarchical_sim_dict"]]
                ax3.plot(values, label=level, linewidth=2)
            # 平均相似度
            avg_values = [d.get('average', 0) for d in history["hierarchical_sim_dict"]]
            ax3.plot(avg_values, label='Average', linewidth=2, linestyle='--', color='black')
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Cosine Similarity")
            ax3.set_title("Hierarchical Intra-Cluster Similarity")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No similarity data", ha='center', va='center')
            ax3.set_title("Hierarchical Similarity")

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
    def visualize_embeddings_by_soc(
        level_embeddings: Dict[str, pd.DataFrame],
        processor: HeteroMedDRADataProcessor,
        sample_size: int = 3000,
        method: str = "tsne",
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        按 SOC 着色可视化嵌入，并打印各 SOC 的簇内平均余弦相似度。
        """
        if not level_embeddings:
            logger.warning("无嵌入数据可供可视化")
            return None

        logger.info(f"使用 {method.upper()} 按 SOC 可视化嵌入并计算余弦相似度...")

        # 收集嵌入和 SOC 标签
        all_embs = []
        all_soc_labels = []
        level_names = [l for l in level_embeddings.keys() if l != "all"]
        for level in level_names:
            df = level_embeddings[level]
            dim_cols = [c for c in df.columns if c.startswith("dim_")]
            if not dim_cols or len(df) == 0:
                continue
            embs = df[dim_cols].values
            for idx, node_code in enumerate(df.index):
                soc = processor.get_soc_node_code(node_code)
                if soc is not None and soc >= 0:
                    all_embs.append(embs[idx])
                    all_soc_labels.append(soc)

        if not all_embs:
            logger.warning("未找到有效的 SOC 标签")
            return None

        all_embs = np.array(all_embs)
        all_soc_labels = np.array(all_soc_labels)

        # 分层采样
        from collections import defaultdict
        soc_to_indices = defaultdict(list)
        for i, soc in enumerate(all_soc_labels):
            soc_to_indices[soc].append(i)

        sampled_indices = []
        total_samples = min(sample_size, len(all_embs))
        soc_counts = {soc: len(indices) for soc, indices in soc_to_indices.items()}
        total_count = sum(soc_counts.values())
        for soc, indices in soc_to_indices.items():
            n_avail = len(indices)
            n_sample = max(1, int(total_samples * (n_avail / total_count)))
            n_sample = min(n_sample, n_avail)
            if n_sample > 0:
                sampled_indices.extend(np.random.choice(indices, n_sample, replace=False))

        sampled_embs = all_embs[sampled_indices]
        sampled_labels = all_soc_labels[sampled_indices]

        # 降维
        if method.lower() == "tsne":
            perplexity = min(50, max(5, len(sampled_embs) // 10))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        else:
            reducer = PCA(n_components=2, random_state=42)
        emb_2d = reducer.fit_transform(sampled_embs)

        # 计算簇内平均余弦相似度（基于原始嵌入）
        unique_socs = np.unique(sampled_labels)
        intra_sims = {}
        for soc in unique_socs:
            mask = sampled_labels == soc
            if mask.sum() > 1:
                soc_embs = sampled_embs[mask]
                sim_mat = cosine_similarity(soc_embs)
                triu = np.triu_indices_from(sim_mat, k=1)
                if triu[0].size > 0:
                    intra_sims[soc] = np.mean(sim_mat[triu])
                else:
                    intra_sims[soc] = 1.0
            else:
                intra_sims[soc] = 1.0

        logger.info("各 SOC 簇内平均余弦相似度:")
        for soc, sim in sorted(intra_sims.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {processor.get_node_name(soc)}: {sim:.4f}")

        # 可视化
        plt.figure(figsize=(14, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_socs)))
        for i, soc in enumerate(unique_socs):
            mask = sampled_labels == soc
            if mask.sum() > 0:
                soc_name = processor.get_node_name(soc)
                plt.scatter(
                    emb_2d[mask, 0],
                    emb_2d[mask, 1],
                    c=[colors[i]],
                    label=f"{soc_name} ({mask.sum()})",
                    alpha=0.6,
                    s=15,
                )

        avg_intra = np.mean(list(intra_sims.values()))
        plt.title(f"MedDRA Embedding by SOC ({method.upper()})\n"
                  f"Avg Intra-SOC Cosine Similarity: {avg_intra:.4f}",
                  fontsize=16, fontweight="bold")
        plt.xlabel(f"{method.upper()} Dimension 1")
        plt.ylabel(f"{method.upper()} Dimension 2")
        plt.legend(loc="best", fontsize=8, ncol=3)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"可视化图已保存: {save_path}")
        plt.show()

        return emb_2d

# =============================================================================
# 7. 模型导出工具
# =============================================================================

class ModelExporter:
    @staticmethod
    def save_embeddings(level_embeddings: Dict[str, pd.DataFrame], output_prefix: str = "meddra_noGNN_embeddings") -> None:
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
    def save_model(model: HeteroMedDRAEncoder, filepath: str = "best_meddra_encoder_noGNN.pth") -> None:
        torch.save(model.state_dict(), filepath)
        logger.info(f"✅ 模型权重已保存: {filepath}")

    @staticmethod
    def save_checkpoint(
        model: HeteroMedDRAEncoder,
        processor: HeteroMedDRADataProcessor,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        best_metrics: Dict,
        filepath: str = "best_checkpoint_noGNN.pt",
    ) -> None:
        """保存完整 checkpoint，包含模型、配置、元数据。"""
        from dataclasses import asdict

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "node_types": list(model.node_types),
            "num_socs": model.num_socs,
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "level_map": processor.level_map,
            "level_names": processor.level_names,
            "level_nodes": {k: list(v) for k, v in processor.level_nodes.items()},
            "soc_list": processor.level_nodes.get("SOC", []),
            "num_all_nodes": processor.num_all_nodes,
            "best_metrics": best_metrics,
            "cache_version": GraphCacheManager.CACHE_VERSION,
            "build_hierarchy_version": BUILD_HIERARCHY_VERSION,   # 【版本差异检测】
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"✅ 完整 checkpoint 已保存: {filepath}")

    @staticmethod
    # 【预留·未接线】**完整 checkpoint 加载**接口（供推理 / 增量训练用），训练流程未调用；
    # 本文件无 load_model（noGNN 不含该静态方法）。
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

        valid_fields = {f.name for f in fields(ModelConfig)}
        model_config = ModelConfig(**{
            k: v for k, v in ckpt["model_config"].items()
            if k in valid_fields
        })

        model = HeteroMedDRAEncoder(
            node_types=ckpt["node_types"],
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
# 8. 特征缓存工具函数（不变）
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
    feature_type: str = "MedBERT",
    force_rebuild: bool = False,
    verbose: bool = True,
    feature_model: str = "PubMedBERT",
    model_local_dir: str = "",
) -> Tuple[Dict[str, torch.Tensor], int]:
    if feature_type == "random":
        if verbose:
            logger.info("📝 生成随机特征（消融实验基线）...")
        features_dict = {}
        feature_dim = 128
        for ntype in processor.level_names_lower:
            nodes = processor.level_nodes.get(ntype.upper(), [])
            if not nodes:
                continue
            features = []
            level_idx = processor.level_map.get(ntype.upper(), 0)
            one_hot = torch.zeros(len(processor.level_map))
            one_hot[level_idx] = 1
            for _ in nodes:
                feature = one_hot.repeat(feature_dim // len(processor.level_map) + 1)[:feature_dim]
                feature = feature + 0.1 * torch.randn(feature_dim)
                feature = feature / (feature.norm() + 1e-8)
                features.append(feature)
            features_dict[ntype] = torch.stack(features)
            if verbose:
                logger.info(f"  {ntype}: {features_dict[ntype].shape} (随机特征)")
        return features_dict, feature_dim

    if not force_rebuild and cache_manager.has_pubmedbert_cache(
        processor.meddra_csv_path, embedding_dim, feature_model
    ):
        if verbose:
            logger.info(f"发现 {feature_model} 特征缓存，正在加载...")
        cached_features = cache_manager.load_pubmedbert_features(
            processor.meddra_csv_path, embedding_dim, feature_model
        )
        if cached_features is not None:
            all_match = True
            for ntype in processor.level_names_lower:
                if ntype in cached_features and hasattr(processor.graph_data[ntype], 'x'):
                    if cached_features[ntype].shape[0] != processor.graph_data[ntype].x.shape[0]:
                        all_match = False
                        logger.warning(f"{ntype} 缓存节点数 ({cached_features[ntype].shape[0]}) "
                                       f"与图数据 ({processor.graph_data[ntype].x.shape[0]}) 不匹配，将重新生成")
                        break
            if all_match:
                return cached_features, 768

    if verbose:
        logger.info(f"📝 生成 {feature_model} 语义特征...")

    # 各文本编码器对应的 HF Hub 名称（若 text_model_dir 已就绪则完全不联网）
    _HF_MODEL_NAMES = {
        "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "SapBERT": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "CODER": "GanjinZero/coder_eng",
    }
    if feature_model not in _HF_MODEL_NAMES:
        logger.error(f"未知 feature_model: {feature_model}，可选 PubMedBERT / SapBERT / CODER")
        return {}, 0

    try:
        # 【本地优先】model_local_dir 非空时离线读本地；目录无权重则下载一次并落盘
        tokenizer, text_encoder = load_text_encoder(
            _HF_MODEL_NAMES[feature_model],
            local_dir=model_local_dir,
            verbose=verbose,
        )
        text_encoder.to(DEVICE)
        text_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False
    except Exception as e:
        logger.error(f"加载 {feature_model} 失败: {e}")
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
                    outputs = text_encoder(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :]
                all_embs.append(cls_emb.cpu())
            node_embs = torch.cat(all_embs, dim=0)
            features_dict[ntype] = node_embs
            if verbose:
                logger.info(f"  {ntype}: {node_embs.shape}")

    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    cache_manager.save_pubmedbert_features(
        processor.meddra_csv_path, features_dict, embedding_dim, feature_model
    )
    return features_dict, 768

# =============================================================================
# 9. 训练入口函数
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
    training_config: Optional[TrainingConfig] = None,
    verbose: bool = True,
) -> Tuple[Optional[HeteroMedDRAEncoder], Optional[HeteroMedDRADataProcessor],
           Optional[Dict], Optional[Dict[str, pd.DataFrame]]]:

    if data_config is None:
        data_config = DataConfig()
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # ---- 0. 运行目录与共享缓存（每实验一个文件夹）----
    # 切到本实验输出目录：此后所有相对路径产物（PNG / 嵌入 CSV / 权重 .pth 等）都落在
    # RUN_DIR 内，实验之间互不覆盖；图/特征缓存固定用共享绝对路径，不随实验分裂。
    os.chdir(RUN_DIR)
    data_config.cache_dir = CACHE_DIR
    # 【目录创建放在这里】而不是 import 时：参数写错（如只写 `--jobs 2` 未给实验名）会在
    # 进入训练前报错退出，此时磁盘上不会留下任何空目录；启动器模式的父进程也不建目录。
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # 【路径类配置的环境变量覆盖】便于"一份脚本跑完所有实验"，无需为每个实验复制脚本改路径
    if os.environ.get("MEDDRA_CSV_PATH"):
        data_config.meddra_csv_path = os.environ["MEDDRA_CSV_PATH"]
    if os.environ.get("MEDDRA_TEXT_MODEL_DIR"):
        data_config.text_model_dir = os.environ["MEDDRA_TEXT_MODEL_DIR"]
    # 【相对路径一律按"启动目录"解析成绝对路径】后面的 os.chdir(RUN_DIR) 会让相对路径指向别处，
    #   故这里一次性固化；与 chdir 的先后顺序无关。
    for _attr in ("meddra_csv_path", "text_model_dir"):
        _v = getattr(data_config, _attr, "")
        if _v and not os.path.isabs(_v):
            setattr(data_config, _attr, os.path.join(_CWD, _v))
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

    # 【并发锁】同一实验目录只允许一个进程：防止两个进程共写 resume.pt / 互相覆盖产物
    if not _acquire_run_lock():
        return None, None, None, None

    # 【新增】全局种子（与 SAGE/GAT 一致）
    # 【2026-10-29 多 seed】环境变量覆盖（不改代码即可换训练种子；留出集不受影响）
    if os.environ.get("MEDDRA_SPLIT_SEED"):
        training_config.split_seed = int(os.environ["MEDDRA_SPLIT_SEED"])
    if os.environ.get("MEDDRA_TRAIN_SEED"):
        training_config.train_seed = int(os.environ["MEDDRA_TRAIN_SEED"])
    _set_seed(training_config._train_seed)
    logger.info(
        f"随机种子: 留出 INDUCTIVE_SEED={INDUCTIVE_SEED}、模型选择集 split_seed={training_config._split_seed}、"
        f"训练 train_seed={training_config._train_seed}；模型选择协议 MEDDRA_IND_SELECT="
        f"{'disjoint' if INDUCTIVE_SELECT_DISJOINT else 'holdout'}"
    )

    logger.info("=" * 60)
    logger.info("🏥 MedDRA 训练模式 [noGNN 基线]")
    logger.info("=" * 60)
    if verbose:
        logger.info("训练配置:")
        logger.info(f"  数据路径: {data_config.meddra_csv_path}")
        logger.info(f"  特征模型: {model_config.feature_model}")
        logger.info(f"  特征类型: {data_config.feature_type}")
        logger.info(f"  嵌入维度: {model_config.embedding_dim}")
        logger.info(f"  训练轮数: {training_config.epochs}")
        logger.info(f"  学习率: {training_config.lr}")
        logger.info(f"  实验名/配置键: {EXPERIMENT}")
        logger.info(f"  实验输出目录: {RUN_DIR}")
        logger.info(f"  缓存目录: {CACHE_DIR}")

    cache_manager = GraphCacheManager(data_config.cache_dir)
    processor = HeteroMedDRADataProcessor(data_config.meddra_csv_path, cache_manager)
    graph_data, _ = processor.load_or_build_graph(
        embedding_dim=model_config.embedding_dim,
        force_rebuild=data_config.force_rebuild_cache,
        verbose=verbose,
    )
    if graph_data is None:
        logger.error("图加载失败，程序退出")
        return None, None, None, None                # 【修复】4 元组

    # ---- 1.5 【归纳式】留出整棵 HLT 子树（noGNN 无消息传递 → 不需要对边做掩码）----
    inductive_holdout: Set[int] = set()
    if INDUCTIVE_MODE:
        inductive_holdout = select_inductive_holdout(processor)
        _manifest = {
            "protocol": "inductive_holdout_hlt_subtree",
            "keep_edges": bool(INDUCTIVE_KEEP_EDGES),
            "hlt_ratio": INDUCTIVE_HLT_RATIO,
            "seed": INDUCTIVE_SEED,
            "selection_protocol": "disjoint" if INDUCTIVE_SELECT_DISJOINT else "holdout",
            "train_seed": int(training_config._train_seed),
            "split_seed": int(training_config._split_seed),
            "num_holdout_nodes": len(inductive_holdout),
            "holdout_node_codes": sorted(inductive_holdout),
            "edge_masking": ("keep_edges (v2; 对 noGNN 无影响)" if INDUCTIVE_KEEP_EDGES
                             else "not_applicable (no message passing)"),
        }
        _mf = os.path.join(RUN_DIR, "inductive_holdout.json")
        with open(_mf, "w", encoding="utf-8") as f:
            json.dump(_manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"【归纳式】留出节点 {len(inductive_holdout):,} 个；清单已写入 {_mf}")
        logger.info("【归纳式】noGNN 不做边掩码：其结果只受划分（训练集变小）影响")

    if verbose:
        logger.info(f"图数据加载成功: {len(processor.node_codes):,} 个节点")

    features_dict, feature_dim = get_pubmedbert_features(
        processor=processor,
        cache_manager=cache_manager,
        embedding_dim=model_config.embedding_dim,
        force_rebuild=data_config.force_rebuild_features,
        feature_type=data_config.feature_type,
        verbose=verbose,
        feature_model=model_config.feature_model,
        model_local_dir=data_config.text_model_dir,
    )
    if not features_dict:
        logger.error("特征生成失败")
        return None, None, None, None                # 【修复】4 元组

    for ntype, features in features_dict.items():
        if hasattr(graph_data[ntype], "x"):
            graph_data[ntype].x = features

    model_config.in_channels = feature_dim
    num_socs = len(processor.level_nodes.get("SOC", []))
    model = HeteroMedDRAEncoder(
        node_types=list(graph_data.node_types),
        num_socs=num_socs,
        model_config=model_config,
    ).to(DEVICE)

    if verbose:
        logger.info(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    trainer = HeteroMedDRATrainer(
        model=model,
        data=graph_data,
        processor=processor,
        training_config=training_config,
        inductive_holdout=inductive_holdout,   # 【归纳式】留出集（转导式传空集）
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
    trainer.es_gap_eps = 1e-4             # 容差带内择优所需的最小 gap 提升
    trainer.es_min_epochs = 60            # 早停保护期：前 60 轮只记录、不累计 patience（≈200轮的30%）
    trainer.es_loss_min_delta = 1e-4      # 验证损失“仍在下降”的判定阈值
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
    #      说明：本文件无对比损失与图消息传递，contrastive_config 传 None，
    #            横幅会明确打印“对比学习: 关闭”。
    log_run_banner(
        model_kind="NoGNN（纯文本基线：无图消息传递、无对比学习）",
        aggregation="无（各节点类型独立线性投影，不做任何邻域聚合）",
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        contrastive_config=None,
        graph_data=graph_data,
        processor=processor,
        verbose=verbose,
    )

    trained_model, history = trainer.train(verbose=verbose)

    # ---- 生成实验标识，避免消融实验互相覆盖（v7.1-noGNN 修复）----
    # 说明：tag 同时包含 feature_model 与 feature_type，
    #       确保 PubMedBERT/SapBERT/CODER × MedBERT/random 的产物不互相覆盖。
    experiment_tag = (
        f"noGNN_{model_config.feature_model}_"
        f"{data_config.feature_type}"
    )
    logger.info(f"📌 实验标识: {experiment_tag}")

    # 【隔离自检】最终产物一律写进本实验目录：PNG / CSV / 权重 / checkpoint 都用相对路径保存，
    # 一旦 cwd 不是 RUN_DIR 就会落错地方（实验之间互相污染）→ 这里强制纠正回来。
    if os.path.abspath(os.getcwd()) != os.path.abspath(RUN_DIR):
        logger.warning(f"⚠️ 当前工作目录不是实验目录（{os.getcwd()}）→ 已纠正为 {RUN_DIR}")
        os.chdir(RUN_DIR)

    cache_manager.save_train_history(
        data_config.meddra_csv_path,
        model_config.embedding_dim,
        history,
        tag=experiment_tag,
        out_dir=RUN_DIR,
    )

    level_embeddings = trainer.generate_embeddings()
    if level_embeddings:
        ModelExporter.save_embeddings(
            level_embeddings,
            output_prefix=f"meddra_noGNN_embeddings_{experiment_tag}",
        )
        ModelExporter.save_model(
            trained_model,
            f"best_meddra_encoder_noGNN_{experiment_tag}.pth",
        )
        # 【新增】保存完整 checkpoint
        ModelExporter.save_checkpoint(
            model=trained_model,
            processor=processor,
            model_config=model_config,
            training_config=training_config,
            best_metrics=trainer.best_metrics,
            filepath=f"best_checkpoint_noGNN_{experiment_tag}.pt",
        )

    # 可视化：**全部容错**（绘图失败只告警；权重/指标/嵌入此前已落盘）
    _safe_visual(Visualizer.plot_training_history, history, model_config.feature_model,
                 save_path=f"training_history_noGNN_{experiment_tag}.png")

    if level_embeddings:
        _safe_visual(Visualizer.visualize_embeddings_by_soc, level_embeddings, processor,
                     sample_size=3000, method="tsne",
                     save_path=f"embeddings_by_soc_noGNN_{experiment_tag}.png")

    device_manager.clear_memory()

    logger.info("\n🎉 训练完成！ [noGNN 基线]")
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
    logger.info(f"📌 实验标识: {experiment_tag}")

    return trained_model, processor, history, level_embeddings


# =============================================================================
# 10. 主程序（无命令行，直接运行）
# =============================================================================

# =============================================================================
# 命令行启动器：python <本脚本> IND-PubMedBert IND-SapBERT [--jobs N] [--dry-run]
# -----------------------------------------------------------------------------
# 机制（命令行模式只当"启动器"，自己不训练）：
#   ① 为每个实验名创建/复用目录 <结果根>/<实验名>/（已存在则复用 → 断点续训可直接接着跑）
#   ② 把本脚本复制一份进去（版本留档：该目录里的脚本就是产生该结果的代码）
#   ③ 用**独立子进程**在各自目录里运行它（子进程互不共享状态，避免显存/缓存串味）
#   ④ 用环境变量把实验名与输出目录传给子进程，子进程无需再解析参数
# 结果根目录 = $MEDDRA_OUTPUT_ROOT，未设置则取**当前工作目录**。
# 显存提醒：NoGNN 约 3–4 GB/个，可 2–3 并发。
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
    """解析 `IND-PubMedBert IND-SapBERT --jobs 2 --dry-run`，返回 (实验名列表, 并发数, dry-run)。"""
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
        logger.error("或把本脚本保存成 .py 文件后再用：python <脚本>.py IND-PubMedBert [--jobs N]")
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
    # 实验注册表：实验名（= RUN_DIR 文件夹名）→ 相对默认配置的覆盖项。
    # 三者只差文本编码器；特征缓存文件名已含 feature_model，互不覆盖。
    #   $env:MEDDRA_EXPERIMENT='NoGNN-SapBERT'; python meddra_full_trainer_noGNN.py
    _NOGNN_EXPERIMENTS = {
        # 无消息传递 → 掩蔽/保留层级边对结果没有影响，故不需要 KEEP 变体：
        # 一次运行同时充当两个协议下的无图基线（与旧跑保持一致：归纳式 NOGNN 只跑一次）
        "IND-NOGNN-PUBMEDBERT": {"keep_edges": False, "feature_model": "PubMedBERT"},  # 必跑
        "IND-NOGNN-SAPBERT":    {"keep_edges": False, "feature_model": "SapBERT"},     # 可选
        "IND-NOGNN-CODER":      {"keep_edges": False, "feature_model": "CODER"},       # 可选
        # 旧名兼容
        "IND-PubMedBert": {"keep_edges": False, "feature_model": "PubMedBERT"},
        "IND-SapBERT":    {"keep_edges": False, "feature_model": "SapBERT"},
        "IND-CODER":      {"keep_edges": False, "feature_model": "CODER"},
    }

    # ---- 命令行给了实验名 → 启动器模式：每个实验自动跑在自己文件夹里 ----
    _names, _jobs, _dry = _parse_cli(sys.argv[1:])
    # 【参数预检】给了参数却解析不出实验名（如只写 `--jobs 2`）→ 直接报错并给用法，
    # 而不是"静默按默认配置开始训练"（那样会跑成另一个实验，且不易察觉）。
    # notebook/单元格下 _SCRIPT_PATH 为空 → 跳过（那里的 sys.argv 是内核参数，不是用户参数）。
    _user_args = [a for a in sys.argv[1:] if _SCRIPT_PATH and not a.startswith("-")]
    if _user_args and not _names:
        logger.error(f"参数里没有可识别的实验名: {sys.argv[1:]}；"
                     f"用法: python <脚本>.py PubMedBert SapBERT [--jobs 2] [--dry-run]")
        raise SystemExit(2)
    if _names:
        raise SystemExit(_launch_experiments(_names, _NOGNN_EXPERIMENTS, jobs=_jobs, dry_run=_dry))

    # 实验名解析：精确命中 → 等价命名互换（PubMedBert ↔ IND-PubMedBert）→ 默认配置（并告警）。
    _key = _resolve_experiment(EXPERIMENT, _NOGNN_EXPERIMENTS, "IND-PubMedBert")
    _ov = _NOGNN_EXPERIMENTS[_key]
    # 【v1/v2 协议】注册表里的 keep_edges 覆盖模块常量；MEDDRA_KEEP_EDGES 环境变量优先级最高
    if "keep_edges" in _ov:
        globals()["INDUCTIVE_KEEP_EDGES"] = bool(_ov["keep_edges"])
    if os.environ.get("MEDDRA_KEEP_EDGES") is not None:
        globals()["INDUCTIVE_KEEP_EDGES"] = str(os.environ["MEDDRA_KEEP_EDGES"]).strip().lower() not in (
            "", "0", "false", "off")
    logger.info(
        "归纳式协议: INDUCTIVE_KEEP_EDGES=%s（False=EDGE-MASK 掩蔽留出子树层级边 / True=EDGE-KEEP 保留边）",
        INDUCTIVE_KEEP_EDGES)

    _model_cfg = ModelConfig()
    if "feature_model" in _ov:
        _model_cfg.feature_model = _ov["feature_model"]

    # ---- 执行全量训练 ----
    trained_model, processor, history, level_embeddings = run_full_training(
        model_config=_model_cfg,
    )

    # 训练未正常完成（并发锁冲突 / 图加载失败等）→ 非零退出码，便于启动器与外部脚本识别失败
    if trained_model is None:
        raise SystemExit(1)
