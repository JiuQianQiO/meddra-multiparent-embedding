# -*- coding: utf-8 -*-
"""B2: knowledge-graph embedding (KGE) baseline: TransE / RotatE / ComplEx (PyKEEN).

Question answered: "what is the ceiling of pure structure, without any text semantics?"
    -- complementary to V3 (random features + GNN).

Usage (paths and hyperparameters are set centrally in config.py; all command-line arguments are **optional**):
    python meddra/baseline/train_b2_kge.py                          # uses B2_MODEL from config.py
    python meddra/baseline/train_b2_kge.py --model RotatE           # specify only to switch models temporarily
Dependencies: pip install pykeen (a dedicated virtual environment is recommended to avoid
    conflicts with PyG).
Notes: the training objective is link prediction (consistent with the KGE literature), whereas
    evaluation is not link prediction -- after training the entity embeddings are **frozen**
    and SOC classification is performed with the same linear probe as in B1.
"""

# --------------------------------------------------------------------
# 内联公共设施（原 baseline_common.py）——本脚本自带全部公共代码，可直接单独运行：
#   划分逐行复刻主实验逻辑 + 强制对账 / 统一线性探针 / 结果 JSON / 资源台账
# --------------------------------------------------------------------
import datetime
import importlib.util
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import hashlib
import logging
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

SEED = 42
def _script_dir() -> str:
    """脚本所在目录。在 Jupyter / Kaggle Notebook 中 `__file__` 未定义，退回当前工作目录。"""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


BASE_DIR = _script_dir()
REPO = os.path.dirname(os.path.dirname(BASE_DIR))


# ==================== 配置区：单文件独立运行时，直接修改这里即可 ====================
# 覆盖优先级：BaselineConfig(...) 传参 > 环境变量 > 同目录 config.py（可选） > 下列默认值
DEFAULT_CSV_PATH = r"/path/to/MEDDRA_DATA_29_0.csv"   # MedDRA 官方层级 CSV
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "meddra_cache")      # 图 / 特征缓存目录
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "baseline_outputs")    # 结果输出目录（自动创建）
DEFAULT_SEED = 42
DEFAULT_DEVICE = "auto"                                         # "auto" → 有 CUDA 用 cuda
B2_MODELS = ["TransE", "RotatE", "ComplEx"]  # 依次跑哪些 KGE 模型；只跑一个写 ["TransE"]
B2_DIM = 256
B2_EPOCHS = 100
B2_BATCH_SIZE = 1024



def _load_config() -> dict:
    """读取**可选**配置文件 config.py（脚本同目录 → 当前工作目录）。
    优先级：BaselineConfig 传参 > 环境变量 > config.py > 脚本顶部配置区；
    config.py 不存在时直接用配置区默认值，不报错。"""
    defaults = {
        "CSV_PATH": DEFAULT_CSV_PATH, "CACHE_DIR": DEFAULT_CACHE_DIR, "OUT_DIR": DEFAULT_OUT_DIR,
        "SEED": DEFAULT_SEED, "DEVICE": DEFAULT_DEVICE,
        "B2_MODELS": B2_MODELS, "B2_DIM": B2_DIM, "B2_EPOCHS": B2_EPOCHS,
        "B2_BATCH_SIZE": B2_BATCH_SIZE,
    }
    cands = [os.path.join(BASE_DIR, "config.py"), os.path.join(os.getcwd(), "config.py")]
    for cand in cands:
        if not os.path.isfile(cand):
            continue
        spec = importlib.util.spec_from_file_location("_baseline_cfg", cand)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k in list(defaults):
            v = getattr(mod, k, None)
            if v not in (None, ""):
                defaults[k] = v
        break
    return defaults


_CFG = _load_config()


# -------- 运行期路径：取 config.py / 环境变量的覆盖，否则用上方配置区默认值 --------
CSV_PATH = _CFG["CSV_PATH"]                                                  # MedDRA 官方层级 CSV
CACHE_DIR = _CFG["CACHE_DIR"] or os.path.join(BASE_DIR, "meddra_cache")      # 图 / 特征缓存目录
OUT_DIR = _CFG["OUT_DIR"] or os.path.join(BASE_DIR, "baseline_outputs")      # 结果输出目录
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:                      # 只读目录（部分平台）→ 退回当前工作目录
    OUT_DIR = os.getcwd()
LEDGER = os.path.join(OUT_DIR, "baseline_resources.csv")
ACTIVE_OUT_DIR = OUT_DIR      # 运行期输出目录；main 中由 cfg.out_dir 覆盖
VAL_BY_LEVEL = {"hlgt": 34, "hlt": 174, "pt": 2742, "llt": 9091}   # 逐层验证样本量对账值
DEFAULT_CSV = CSV_PATH


@dataclass
class BaselineConfig:
    """统一配置类：**路径与超参的唯一入口**，不使用任何命令行参数。

    取值优先级：显式传参 `BaselineConfig(...)` > 环境变量 > 同目录 `config.py` > 内置默认值。
    单文件自包含：把本脚本（可连同 config.py）复制到任意机器 / Kaggle，按需修改 config.py 即可。
    """
    # ---- 路径 ----
    csv_path: str = os.environ.get("MEDDRA_CSV_PATH", _CFG["CSV_PATH"])
    cache_dir: str = os.environ.get("MEDDRA_CACHE_DIR", _CFG["CACHE_DIR"] or CACHE_DIR)
    out_dir: str = _CFG["OUT_DIR"] or OUT_DIR
    # ---- 通用 ----
    seed: int = _CFG["SEED"]
    device: str = _CFG["DEVICE"]          # "auto" → 有 CUDA 用 cuda，否则 cpu
    probe_C: tuple = (0.25, 1.0, 4.0)     # 线性探针超参搜索范围
        # ---- B2 专用 ----
    models: tuple = tuple(_CFG["B2_MODELS"])  # 依次运行的 KGE 模型
    dim: int = _CFG["B2_DIM"]
    epochs: int = _CFG["B2_EPOCHS"]
    batch_size: int = _CFG["B2_BATCH_SIZE"]

    def __post_init__(self):
        self.device = str(DeviceManager(self.device).device)   # auto / CUDA 不可用时自动降级为 CPU
        if isinstance(self.models, str):
            self.models = (self.models,)
        bad = [m for m in self.models if m not in ("TransE", "RotatE", "ComplEx")]
        if bad:
            raise ValueError(f"B2_MODELS 含未知模型 {bad}，可选值: TransE / RotatE / ComplEx")
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except OSError:                   # 只读目录（部分平台）→ 退回当前工作目录
            self.out_dir = os.getcwd()


class DeviceManager:
    """轻量设备管理器（契约对齐主实验 `Meddra_full_trainer_SAGE.py::DeviceManager`，
    为单文件独立运行而内联实现）。

    - `preferred="auto"`：有 CUDA 用 `cuda`，否则 `cpu`；
    - 显式指定 `cuda` / `cuda:1` 但 CUDA 不可用时：**降级为 CPU** 并打印原因，避免直接崩溃；
    - `info()`：打印设备名与显存总量（便于在服务器 / Kaggle 上确认真的用上 GPU）；
    - `clear_memory()`：释放 CUDA 缓存。
    """

    def __init__(self, preferred: str = "auto"):
        self.preferred = str(preferred or "auto")
        self.device = self._resolve(self.preferred)
        self.device_type = str(self.device).split(":")[0]      # cuda / cpu（autocast 用）

    @staticmethod
    def _resolve(pref: str) -> torch.device:
        p = str(pref).strip().lower()
        if p in ("", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if p.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"[DeviceManager] 请求设备 {pref!r}，但 CUDA 不可用 → 降级为 CPU")
                return torch.device("cpu")
            return torch.device(p)
        return torch.device(p)

    def info(self) -> str:
        if self.device.type == "cuda":
            i = self.device.index or 0
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            desc = f"CUDA:{i} {name}（显存 {mem:.2f} GB）"
        else:
            desc = "CPU（未使用 GPU；B1–B4 在 CPU 上耗时可能达数十小时）"
        print(f"[DeviceManager] 运行设备 = {desc}")
        return desc

    def clear_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __str__(self) -> str:
        return str(self.device)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# ---------------------------------------------------------------------------
# 内联数据层（逐字取自 Embedding/code/Base/Meddra_full_trainer_SAGE.py，未作任何修改）：
#   BUILD_HIERARCHY_VERSION（第 272 行）
#   GraphCacheManager（第 602–980 行）
#   HeteroMedDRADataProcessor（第 987–1707 行）
# 目的：本脚本**单文件即可运行**，不导入项目内任何其他脚本；
#       与主实验共用同一套建图 / 缓存 / SOC 标签逻辑，请勿就地改动。
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
BUILD_HIERARCHY_VERSION = "v7.2"

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
        构建父节点缓存（用于对比学习中的父子/叔父关系）。
        
        多父节点处理说明：
            MedDRA 是 DAG 结构，存在显著多父现象：
            - HLGT: 17/337 (5.04%) 有 2 个 SOC 父节点
            - HLT:  18/1739 (1.04%) 有 2 个 HLGT 父节点
            - PT:   10350/27361 (37.83%) 有 2~7 个 HLT 父节点
            - LLT:  0/91082 (0%) 唯一父节点
            
            对比学习的父子正样本和叔父硬负样本需要唯一父节点。
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


def _resolve_csv(cfg: "BaselineConfig") -> str:
    """解析 MedDRA CSV 实际路径（跨平台）：

    依次尝试：① 配置给定的 csv_path；② Kaggle 输入目录（<DATA_DIR>
    ③ 当前工作目录及其子目录；④ 脚本配置区的内置默认路径。
    找到的第一个存在的文件即为实际路径，并打印提示；全部找不到时给出可操作的报错。
    """
    import glob as _g
    cands = [cfg.csv_path]
    for pat in ("<DATA_DIR>",
                os.path.join(os.getcwd(), "**", "MEDDRA_DATA_29_0.csv")):
        try:
            cands += _g.glob(pat, recursive=True)
        except OSError:
            pass
    cands.append(DEFAULT_CSV_PATH)
    for p in cands:
        if p and os.path.isfile(p):
            if os.path.abspath(p) != os.path.abspath(cfg.csv_path or ""):
                print(f"[配置] CSV_PATH 指向的文件不可用，自动改用: {p}")
            return p
    raise FileNotFoundError(
        "未找到 MedDRA CSV（MEDDRA_DATA_29_0.csv）。请在 config.py / 脚本配置区设置 CSV_PATH，"
        "或在 Kaggle 上把该 CSV 作为数据集挂载到 <DATA_DIR> "
        + ", ".join([str(c) for c in cands[:4]]))


def build_processor(cfg: BaselineConfig):
    """构建数据处理器并加载/复用图缓存；返回 (processor, graph)。

    自包含：直接使用本文件内联的 GraphCacheManager / HeteroMedDRADataProcessor，
    不再导入项目内任何其他脚本（单文件即可运行）。
    """
    cfg.csv_path = _resolve_csv(cfg)      # 跨平台解析 CSV 实际路径
    cm = GraphCacheManager(cfg.cache_dir)
    proc = HeteroMedDRADataProcessor(cfg.csv_path, cm)
    graph, _ = proc.load_or_build_graph(embedding_dim=256, force_rebuild=False, verbose=False)
    if graph is None:
        raise RuntimeError("图加载失败（缓存缺失或版本不匹配）——请检查 config.py 的 CACHE_DIR 与 CSV_PATH")
    return proc, graph


def get_split(proc, graph, seed: int = SEED, val_ratio: float = 0.1,
              test_ratio: float = 0.1) -> dict:
    """逐行复刻 HeteroMedDRATrainer._prepare_training_data 的节点枚举与 SOC 分层划分，
    保证与主实验**逐成员一致**；随后做结构对账（不硬编码具体规模，跨版本亦适用）。

    JBI 投稿起改为三分切分（train/val/test，test 仅最终评估一次）：
      - 枚举：按 graph.node_types 顺序遍历各层 node_codes，跳过 SOC；
        SOC 类别索引 = soc_list（level_nodes['SOC'] 顺序）中主 SOC 代码的位次；
      - 划分：rng = np.random.RandomState(seed)；先按 test_ratio 划测试集，再在剩余里
        按 val_ratio 划验证集，其余为训练集；len<=2 的池（无 test 时）取首个为验证。
      - test_ratio=0 时退化为原二分协议（train/val）。
    """
    soc_list = proc.level_nodes.get("SOC", [])
    soc_to_idx = {soc: i for i, soc in enumerate(soc_list)}
    all_nodes, all_node_types, all_soc_labels = [], [], []
    for nt in graph.node_types:
        for code in graph[nt].node_codes:
            code = int(code)
            level = proc.get_node_level(code)
            if level and level in proc.level_map:
                if level == "SOC":
                    continue
                all_nodes.append(code)
                all_node_types.append(nt)
                soc = proc.get_soc_node_code(code)
                all_soc_labels.append(soc_to_idx[soc] if soc is not None and soc in soc_to_idx else -1)

    soc_to_indices = {}
    for idx, soc_lab in enumerate(all_soc_labels):
        if soc_lab >= 0:
            soc_to_indices.setdefault(soc_lab, []).append(idx)
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for lab, indices in soc_to_indices.items():
        _lst = list(indices)
        # 先划测试集
        if test_ratio > 0 and len(_lst) >= 3:
            n_test = max(1, min(int(len(_lst) * test_ratio), len(_lst) // 2))
            test_sel = rng.choice(_lst, n_test, replace=False)
            _test_set = {int(t) for t in test_sel}
            _rest = [i for i in _lst if i not in _test_set]
        else:
            test_sel, _test_set = [], set()
            _rest = _lst
        # 再划验证集
        n_val = max(1, min(int(len(_rest) * val_ratio), len(_rest) // 2))
        if len(_rest) <= 2:
            val_sel = _rest[:1]
            train_sel = _rest[1:]
        else:
            val_sel = rng.choice(_rest, n_val, replace=False)
            _val_set = {int(v) for v in val_sel}
            train_sel = [i for i in _rest if i not in _val_set]
        test_idx.extend(test_sel)
        val_idx.extend(val_sel)
        train_idx.extend(train_sel)

    tr = np.asarray(train_idx, dtype=np.int64)
    va = np.asarray(val_idx, dtype=np.int64)
    te = np.asarray(test_idx, dtype=np.int64)
    y = np.asarray(all_soc_labels, dtype=np.int64)
    nt = np.asarray(all_node_types)

    # 【结构对账】不再硬编码具体规模（跨版本节点数会变），只校验划分的结构正确性
    _s_tr, _s_va, _s_te = set(map(int, tr)), set(map(int, va)), set(map(int, te))
    if len(_s_tr & _s_va) or len(_s_tr & _s_te) or len(_s_va & _s_te):
        raise AssertionError("train/val/test 三者存在交集")
    if (_s_tr | _s_va | _s_te) != set(range(len(all_nodes))):
        raise AssertionError("train/val/test 并集不等于全集（有遗漏或重复）")
    for lab, indices in soc_to_indices.items():
        _s = set(indices)
        if len(_s & _s_tr) + len(_s & _s_va) + len(_s & _s_te) != len(_s):
            raise AssertionError(f"SOC{lab} 池分层不完整")

    # ---- JBI 投稿：与主线导出的「统一划分索引」逐成员交叉校验 ----
    _sp_file = os.environ.get("MEDDRA_SPLIT_INDEX_FILE") or ""
    if not _sp_file:
        for _c in (os.path.join(BASE_DIR, "meddra_cache", "split_indices.json"),
                   os.path.join(os.path.dirname(BASE_DIR), "meddra_cache", "split_indices.json")):
            if os.path.exists(_c):
                _sp_file = _c
                break
    if _sp_file and os.path.exists(_sp_file):
        import json as _sp_json
        with open(_sp_file, "r", encoding="utf-8") as _sp_f:
            _sp_ref = _sp_json.load(_sp_f)
        if [int(c) for c in _sp_ref.get("nodes", [])] != [int(c) for c in all_nodes]:
            raise AssertionError(f"划分交叉校验失败：{_sp_file} 的节点枚举与本脚本不一致")
        _sp_mine = {"train": set(int(i) for i in tr),
                    "val": set(int(i) for i in va),
                    "test": set(int(i) for i in te)}
        for _k, _s in _sp_mine.items():
            if set(int(i) for i in _sp_ref.get(_k, [])) != _s:
                raise AssertionError(
                    f"划分交叉校验失败：{_sp_file} 的 {_k} 集与本脚本划分不一致"
                    f"（文件 {len(_sp_ref.get(_k, []))} 个，本脚本 {len(_s)} 个）")
        print(f"[划分] 逐成员交叉校验通过 ✓  {_sp_file}")
    else:
        print("[划分] ⚠\ufe0f 未找到统一划分索引，跳过交叉校验；"
              "请设 MEDDRA_SPLIT_INDEX_FILE 指向主线导出的 split_indices.json")

    return {"train": tr, "val": va, "test": te,
            "y": y, "ntype": nt, "codes": np.asarray(all_nodes)}


def compute_metrics(y_true, y_pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4)}


def linear_probe(emb: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                 test_idx: np.ndarray = None, c_list=(0.25, 1.0, 4.0), seed: int = SEED):
    """冻结嵌入 + 线性探针。超参 C 用验证集选择；在测试集（若提供）或验证集上报告。

    JBI 投稿起：C 用 val 集选（而非训练集内部再切），最终指标在 test 集报告（仅评估一次）；
    若 test_idx 为 None，则退化为「val 报告」的旧二分语义。
    """
    best_C, best_acc = None, -1.0
    for c in c_list:
        clf = LogisticRegression(max_iter=2000, C=c)
        clf.fit(emb[train_idx], y[train_idx])
        acc = accuracy_score(y[val_idx], clf.predict(emb[val_idx]))
        if acc > best_acc:
            best_C, best_acc = c, acc
    clf = LogisticRegression(max_iter=2000, C=best_C)
    clf.fit(emb[train_idx], y[train_idx])
    _report_idx = test_idx if test_idx is not None and len(test_idx) > 0 else val_idx
    metrics = compute_metrics(y[_report_idx], clf.predict(emb[_report_idx]))
    metrics["probe_C"] = best_C
    return metrics, clf


def save_results(name: str, metrics: dict, extra: dict | None = None,
                 out_dir: str = None) -> str:
    """保存与 best_metrics 同构的结果 JSON（供扩版表 4-5 与审计脚本读取）。"""
    out_dir = out_dir or ACTIVE_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    payload = {"name": name,
               "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
               "metrics": metrics, "extra": extra or {}}
    p = os.path.join(out_dir, f"result_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"  结果已保存: {p}")
    return p


def append_ledger(name: str, note: str = "", out_dir: str = None) -> None:
    """追加资源台账（与主实验 platform_resources.csv 同格式；基线不走平台监控，故内存/显存/耗时记 '-'）。"""
    line = f"{name},-,-,-,{note}"
    out_dir = out_dir or ACTIVE_OUT_DIR
    ledger = os.path.join(out_dir, "baseline_resources.csv")
    os.makedirs(out_dir, exist_ok=True)
    header_needed = not os.path.exists(ledger)
    with open(ledger, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# 外部基线资源台账（格式与 platform_resources.csv 对齐；'-' 表示平台监控未覆盖）\n")
            f.write("# 配置,内存GB,显存GB,单轮耗时s,实际轮数\n")
        f.write(line + "\n")
    print(f"  台账已记录: {line}")


import os
import sys
import time

import numpy as np
import torch


def build_index_maps(proc, graph, sp: dict) -> np.ndarray:
    """划分枚举索引（120,519，不含 SOC）→ 折叠全局索引（120,546，含 SOC）的映射（详见 B1 同名函数）。"""
    starts, acc, code2fold = {}, 0, {}
    for nt in graph.node_types:
        starts[nt] = acc
        for li, c in enumerate(graph[nt].node_codes):
            code2fold[(nt, int(c))] = acc + li
        acc += len(graph[nt].node_codes)
    return np.asarray([code2fold[(nt, int(c))] for nt, c in zip(sp["ntype"], sp["codes"])],
                      dtype=np.int64)


def _run_one(cfg: BaselineConfig, model_key: str) -> int:
    """主流程；cfg 缺省时使用统一配置类（config.py / 环境变量 / 内置默认值）。"""
    cfg = cfg or BaselineConfig()
    global ACTIVE_OUT_DIR
    ACTIVE_OUT_DIR = cfg.out_dir        # 结果 JSON 与台账写入 cfg.out_dir
    dev = DeviceManager(cfg.device)     # 设备管理器（自动选择 / 降级 / 打印信息）
    dev.info()

    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        print("缺少 pykeen：请先 pip install pykeen（建议独立虚拟环境）")
        return 2

    seed_everything(cfg.seed)
    tag = f"B2_{model_key}"
    t0 = time.time()

    print(f"[1/4] 构建主实验同款处理器与划分（{tag}）")
    proc, graph = build_processor(cfg)
    sp = get_split(proc, graph)
    y, tr, va, te = sp["y"], sp["train"], sp["val"], sp["test"]

    print("[2/4] 三元组构建（四类父子边 + 反向）")
    # 边索引是**层内局部索引**，必须先加该层在折叠坐标中的起始偏移，否则不同层的同名索引会被误合并
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
    tf = TriplesFactory(mapped_triples=mapped, entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    print(f"    三元组 {len(mapped):,}，实体 {len(entities):,}，关系 {len(relations)}")

    emb_p = os.path.join(cfg.out_dir, f"{tag}_embeddings.pt")
    os.makedirs(cfg.out_dir, exist_ok=True)
    reuse = None
    if os.path.isfile(emb_p):                       # 已有实体嵌入 → 直接复用，不再训练
        _ck = torch.load(emb_p, map_location="cpu", weights_only=False)
        if isinstance(_ck, dict) and "emb" in _ck and "entity_to_id" in _ck:
            reuse = _ck
    if reuse is not None:
        emb_all = reuse["emb"].float().numpy()
        entity_to_id = reuse["entity_to_id"]
        print(f"    复用已有实体嵌入缓存（跳过训练）: {emb_p} {tuple(emb_all.shape)}")
    else:
        # 用 SLCWA 训练循环直接训练（不做链接预测评估，仅取实体嵌入）。
        # 不用 pykeen.pipeline()：新版 pipeline 在只给训练集（无 testing）时会直接报错。
        print(f"[3/4] PyKEEN 训练（SLCWA 训练循环）：{model_key}"
              f"（dim={cfg.dim}, epochs={cfg.epochs}, seed={cfg.seed}）")
        import pykeen.models as _pm
        from pykeen.training import SLCWATrainingLoop
        model = getattr(_pm, model_key)(triples_factory=tf, embedding_dim=cfg.dim).to(cfg.device)
        opt = torch.optim.Adam(model.get_grad_params(), lr=1e-3)   # 与 pipeline 默认优化器一致
        loop = SLCWATrainingLoop(model=model, triples_factory=tf, optimizer=opt)
        _losses = loop.train(triples_factory=tf, num_epochs=cfg.epochs,
                             batch_size=cfg.batch_size, use_tqdm=True)
        emb_all = model.entity_representations[0](indices=None).detach().cpu().numpy().astype(np.float32)
        # ComplEx 等模型的实体表示可能是 (n, 2, dim)（实部/虚部），需展平为 (n, dim)
        if emb_all.ndim > 2:
            emb_all = emb_all.reshape(emb_all.shape[0], -1)
            print(f"    实体表示已展平为 {emb_all.shape}（复数/多分量表示）")
        torch.save({"emb": torch.from_numpy(emb_all).half(),
                    "entity_to_id": entity_to_id}, emb_p)
        print(f"    嵌入缓存: {emb_p} {tuple(emb_all.shape)}")
    # PyKEEN 实体 ID = 折叠全局索引字符串（0..120,545）→ 分配数组必须用折叠坐标长度；
    # 不能按 len(proc.node_codes)（93,185，按代码去重）分配，否则实体索引越界。
    fold_n = sum(len(graph[nt].node_codes) for nt in graph.node_types)
    max_eid = max(int(e) for e in entity_to_id)
    assert max_eid < fold_n, f"实体索引 {max_eid} 超出折叠坐标范围 {fold_n}（坐标系不一致）"
    emb = np.zeros((fold_n, emb_all.shape[1]), dtype=np.float32)
    for e, i in entity_to_id.items():
        emb[int(e)] = emb_all[i]
    # 边界处理：未出现在任何三元组中的节点嵌入为 0，若牵涉训练/验证集须报警
    zero = np.where(~emb.any(axis=1))[0]
    if len(zero):
        print(f"    警告：{len(zero):,} 个节点未出现在任何三元组中（嵌入为 0）")

    print("[4/4] 线性探针（冻结实体嵌入，仅四术语层节点）")
    proc2fold = build_index_maps(proc, graph, sp)
    m, _ = linear_probe(emb[proc2fold], y, tr, va, te, c_list=cfg.probe_C, seed=cfg.seed)
    print(f"    探针: {m}")
    save_results(tag, m, extra={"dim": cfg.dim, "epochs": cfg.epochs,
                                "triples": len(mapped), "library": "pykeen"})
    append_ledger(tag, note=f"PyKEEN {model_key}; {cfg.epochs} epochs; 耗时 {time.time() - t0:.0f}s")
    dev.clear_memory()                  # 释放显存缓存
    return 0


def main(cfg: BaselineConfig = None) -> int:
    """一次运行依次跑 B2_MODELS 里的每个 KGE 模型。"""
    cfg = cfg or BaselineConfig()
    print(f"待运行模型: {list(cfg.models)}")
    done = 0
    for mk in cfg.models:
        print(f"\n===== B2 / {mk} =====")
        if _run_one(cfg, mk) == 0:
            done += 1
    print(f"完成（{done}/{len(cfg.models)} 个模型）。")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
