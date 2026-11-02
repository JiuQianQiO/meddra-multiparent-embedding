# -*- coding: utf-8 -*-
"""run_review_experiments_full.py

JBI 投稿增强实验统一调度（SAGE / NOGNN / GAT / 归纳式 SAGE / 归纳式 NoGNN）。
固定同一 split_indices.json；归纳式固定同一留出集（INDUCTIVE_SEED=42）。

【2026-10-29 17:20 最终分组】只有两个组：
  SUBMIT（39 个，约 429 GPU·h）—— 投稿需要跑的全部实验，一次开完，不再有条件组 / 第二轮：
    · SOC 控制 9：E1B_S42–44、SOCDROP_S42–44、E1A_S42–44（近循环；E1A 用来区分名称语义与节点身份）
     · 多种子 19：V1_S43–46、V2_S43–46、C7_S44–46、V3_S43–46、V4_S43–46（seed 42 / C7 seed 43 取第一轮）
    · 归纳式 7（disjoint 模型选择）：IND_KEEP_S42–46、IND_MASK_S42、IND_NOGNN_S42
     · matched baselines 4：NGPB_S43–44、GAT_S43–44（回应“no-graph / matched GNN 仅单次观察”的第二轮审查风险）
   SKIP（61 个）—— 不需要跑：C2–C6 多种子、ORDCL、敏感性扫描、
    IND_NOGNN_S43/S44、GAT_S45、NGPB_S45–46 等（不改变稿件任何一句结论，理由见 REVIEW_V2_RUNBOOK.md §4）。
  ALL = SUBMIT + SKIP（100 个）。旧短名 G1/G2/G3/MIN/REC/COND_*/P0/P1/SENS/SOCCTRL 已停用，使用时会提示。

用法：
  python run_review_experiments_full.py SUBMIT --print-commands   # 打印一窗口一实验命令（39 条）
  python run_review_experiments_full.py SUBMIT --dry-run          # 只看计划 / 哪些会跳过
  python run_review_experiments_full.py V1_S43 --run --jobs 1     # 单个任务

  · 删除 SOCEDGE_S42/43/44：删掉 SOC 全部边后 SOC 收不到也发不出消息，第 1 层后即从
    计算中消失，"保留 SOC 特征"对任何非 SOC 节点都不起作用 → 与 SOCDROP 计算等价。
  · 删除 HARD100_S42/43/44：hard_negative_weight=1.0 时硬负项系数 (w-1)=0，
    损失与 C4（去硬负样本）逐项相同。
  · 归纳式任务（IND_*）：第一轮归纳式的验证集就是留出集（早停与选权重都在留出集上），
    这里统一用 MEDDRA_IND_SELECT=disjoint——模型选择集取自非留出节点，留出集只在选定权重上
    评估一次（holdout_metrics_*.json）。需要 2026-10-29 修订版 Meddra_inductive_SAGE.py /
    Meddra_inductive_noGNN.py（支持 MEDDRA_TRAIN_SEED 与 disjoint 协议），旧版会被拒绝启动。
"""
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINERS = OrderedDict([
    ("SAGE", os.path.join(HERE, "Meddra_full_trainer_SAGE.py")),
    ("NOGNN", os.path.join(HERE, "meddra_full_trainer_noGNN.py")),
    ("GAT", os.path.join(HERE, "meddra_full_trainer_GAT.py")),
    ("IND_SAGE", os.path.join(HERE, "Meddra_inductive_SAGE.py")),
    ("IND_NOGNN", os.path.join(HERE, "Meddra_inductive_noGNN.py")),
])
# 归纳式训练器必须是 2026-10-29 修订版：旧版不读 MEDDRA_TRAIN_SEED（5 个种子会跑成同一个），
# 也没有 disjoint 协议（会在留出集上早停）。启动前按源码标记核对，缺任一标记即拒绝启动。
_IND_REQUIRED_MARKERS = ("MEDDRA_TRAIN_SEED", "MEDDRA_IND_SELECT", "_inductive_disjoint_split",
                         "holdout_metrics_")
_RUNNING = {}

CACHE_WAIT_TIMEOUT = 12 * 3600

_SOC_CONTROL_MARKERS = {
    "MEDDRA_SOC_NAME_PERMUTE": "MEDDRA_SOC_NAME_PERMUTE",
    "MEDDRA_SOC_ZERO": "MEDDRA_SOC_ZERO",
    "MEDDRA_SOC_DROP": "MEDDRA_SOC_DROP",
    "MEDDRA_SOC_EDGE_MASK": "MEDDRA_SOC_EDGE_MASK",
}
# 仅凭“环境变量字符串存在”不够：旧版训练器也会引用这些变量，但使用错误的
# `hasattr(graph_data, "soc")` 判定，导致 E1A/E1B 静默退化成 V1。
# 这些片段只在 2026-10-28 修复后的正确实现中出现，旧版没有。
_SOC_CONTROL_SAFETY_MARKERS = (
    '"soc" in getattr(graph_data, "node_types", ())',
    'if int(_soc_x.shape[1]) != int(feature_dim):',
)


def _check_trainer_file(trainer):
    """启动前校验训练器文件本身没被上传污染，并返回源码文本。

    为什么要查首行：用 `cat > x.py <<'EOF'` 之类的方式粘贴时，终端里的 `clear`
    很容易被粘成第 1 行（`clear# ====...`），Python 能解析成"表达式语句"，
    于是报的不是语法错误而是 `NameError: name 'clear' is not defined`，
    而且是在训练器进程里炸——排查时会误以为是训练代码的问题。
    """
    with open(trainer, "r", encoding="utf-8") as f:
        source = f.read()
    if not source.strip():
        raise SystemExit("训练器文件为空：%s\n请重新上传。" % trainer)
    first = source.splitlines()[0].lstrip("\ufeff").lstrip()
    _ok_prefixes = ("#", '"""', "'''", 'r"""', "r'''")
    if not first.startswith(_ok_prefixes):
        raise SystemExit("训练器首行异常（疑似上传时混入了终端命令）：%r\n请重新上传 %s"
                         % (first[:80], trainer))
    try:
        compile(source, trainer, "exec")
    except SyntaxError as exc:
        raise SystemExit("训练器语法错误（文件可能被截断或污染）：%s\n%s" % (trainer, exc))
    return source


def _verify_review_controls(trainer, spec):
    """任务请求 SOC 控制时，启动前验证训练器源码包含对应开关，防止旧训练器静默跑成 V1。"""
    requested = [marker for key, marker in _SOC_CONTROL_MARKERS.items()
                 if spec["env"].get(key)]
    if requested:
        requested += list(_SOC_CONTROL_SAFETY_MARKERS)
    if spec["model"].startswith("IND_"):
        requested += list(_IND_REQUIRED_MARKERS)
    if not requested:
        return
    source = _check_trainer_file(trainer)
    missing = [marker for marker in requested if marker not in source]
    if missing:
        raise SystemExit("训练器缺少审稿控制开关：%s\n请更新 %s 后重跑。" % (", ".join(missing), trainer))


JOBS = OrderedDict()


def _add_sage(name, experiment, seed, env=None, desc=""):
    JOBS[name] = dict(model="SAGE", experiment=experiment, seed=str(seed),
                      env=dict(env or {}), desc=desc)


def _add_nognn(name, feature, seed, desc=""):
    # PubMedBERT 使用本地权重目录；SapBERT/CODER 必须按各自 HF 模型名加载，
    # 不能让 MEDDRA_TEXT_MODEL_DIR 把三者都指向 pubmedbert。
    _env = {} if feature == "PubMedBert" else {"MEDDRA_TEXT_MODEL_DIR": ""}
    JOBS[name] = dict(model="NOGNN", experiment=feature, seed=str(seed),
                      env=_env, desc=desc)


def _add_gat(name, seed, desc=""):
    JOBS[name] = dict(model="GAT", experiment="GAT", seed=str(seed),
                      env={}, desc=desc)


def _add_ind(name, model, experiment, seed, desc=""):
    # 归纳式：留出集固定（INDUCTIVE_SEED=42，与第一轮逐成员相同），只换训练种子；
    # 模型选择集取自非留出节点（disjoint），留出集只在选定权重上评估一次。
    JOBS[name] = dict(model=model, experiment=experiment, seed=str(seed),
                      env={"MEDDRA_IND_SELECT": "disjoint"}, desc=desc)


# ---- SOC 名称/结构捷径控制：3 个训练种子 ----
for _seed in ("42", "43", "44"):
    _add_sage("E1A_S%s" % _seed, "V1", _seed,
              {"MEDDRA_SOC_NAME_PERMUTE": "42"},
              "E1A train_seed=%s, shuffle SOC name embeddings fixed at 42" % _seed)
    _add_sage("E1B_S%s" % _seed, "V1", _seed,
              {"MEDDRA_SOC_ZERO": "1"},
              "E1B train_seed=%s, zero SOC name embeddings" % _seed)
    _add_sage("SOCDROP_S%s" % _seed, "V1", _seed,
              {"MEDDRA_SOC_DROP": "1"},
              "SOCDROP train_seed=%s, drop SOC nodes/edges" % _seed)
    # SOCEDGE（MEDDRA_SOC_EDGE_MASK=1）已移除：与 SOCDROP 计算等价，见文件头说明。

# ---- 核心多种子：5 种子标准 ----
for _seed in ("43", "44", "45", "46"):
    _add_sage("V1_S%s" % _seed, "V1", _seed, desc="V1 full train_seed=%s" % _seed)
    _add_sage("V2_S%s" % _seed, "V2", _seed, desc="V2 no-CL train_seed=%s" % _seed)
    _add_sage("V3_S%s" % _seed, "V3", _seed, desc="V3 random no-CL train_seed=%s" % _seed)
    _add_sage("V4_S%s" % _seed, "V4", _seed, desc="V4 random +CL train_seed=%s" % _seed)
    _add_sage("C2_S%s" % _seed, "C2", _seed, desc="C2 train_seed=%s" % _seed)
    _add_sage("C3_S%s" % _seed, "C3", _seed, desc="C3 train_seed=%s" % _seed)
    _add_sage("C4_S%s" % _seed, "C4", _seed, desc="C4 train_seed=%s" % _seed)
    _add_sage("C5_S%s" % _seed, "C5", _seed, desc="C5 train_seed=%s" % _seed)
    _add_sage("C6_S%s" % _seed, "C6", _seed, desc="C6 train_seed=%s" % _seed)

for _seed in ("44", "45", "46"):
    _add_sage("C7_S%s" % _seed, "C7", _seed, desc="C7 train_seed=%s" % _seed)

for _seed in ("42", "43", "44", "45", "46"):
    _add_sage("ORDCL_S%s" % _seed, "OrdCL", _seed,
              desc="OrdCL ordinary sibling-only CL train_seed=%s" % _seed)

# ---- 核心对照模型多种子 ----
for _seed in ("43", "44", "45", "46"):
    _add_nognn("NGPB_S%s" % _seed, "PubMedBert", _seed,
               "NoGNN+PubMedBERT train_seed=%s" % _seed)
for _seed in ("43", "44", "45"):
    _add_nognn("NGSB_S%s" % _seed, "SapBERT", _seed,
               "NoGNN+SapBERT train_seed=%s" % _seed)
    _add_nognn("NGCODER_S%s" % _seed, "CODER", _seed,
               "NoGNN+CODER train_seed=%s" % _seed)
    _add_gat("GAT_S%s" % _seed, _seed, "GAT train_seed=%s" % _seed)

# ---- 权重敏感性：审稿点名的 8 个权重跑 3 seeds，tau/CL 单种子 ----
_WEIGHT_SCANS = [
    ("SIB050", {"MEDDRA_SIB_W": "0.5"}, "primary sibling=0.5"),
    ("SIB150", {"MEDDRA_SIB_W": "1.5"}, "primary sibling=1.5"),
    ("SIB2010", {"MEDDRA_SIB2_W": "0.1"}, "secondary sibling=0.1"),
    ("SIB2050", {"MEDDRA_SIB2_W": "0.5"}, "secondary sibling=0.5"),
    ("PAR020", {"MEDDRA_PARENT_W": "0.2"}, "parent=0.2"),
    ("PAR100", {"MEDDRA_PARENT_W": "1.0"}, "parent=1.0"),
    # HARD100（hard-negative=1.0）已移除：数学上等价于 C4，见文件头说明。
    ("HARD250", {"MEDDRA_HARD_W": "2.5"}, "hard-negative=2.5"),
]
for _base, _env, _desc in _WEIGHT_SCANS:
    for _seed in ("42", "43", "44"):
        _add_sage("%s_S%s" % (_base, _seed), "V1", _seed, _env,
                  "%s train_seed=%s" % (_desc, _seed))

_SINGLE_SCANS = [
    ("TAU010", {"MEDDRA_TAU": "0.1"}, "tau=0.1"),
    ("TAU050", {"MEDDRA_TAU": "0.5"}, "tau=0.5"),
    ("CL010", {"MEDDRA_CL_W": "0.1"}, "contrastive_weight=0.1"),
    ("CL050", {"MEDDRA_CL_W": "0.5"}, "contrastive_weight=0.5"),
]
for _name, _env, _desc in _SINGLE_SCANS:
    _add_sage(_name, "V1", "42", _env, _desc)

# ---- 归纳式（2026-10-29 第二轮）：disjoint 模型选择 + 多训练种子 ----
for _seed in ("42", "43", "44", "45", "46"):
    _add_ind("IND_KEEP_S%s" % _seed, "IND_SAGE", "IND-SAGE-KEEP", _seed,
             "IND-SAGE-KEEP disjoint-select train_seed=%s" % _seed)
_add_ind("IND_MASK_S42", "IND_SAGE", "IND-SAGE-MASK", "42",
         "IND-SAGE-MASK disjoint-select train_seed=42")
for _seed in ("42", "43", "44"):
    _add_ind("IND_NOGNN_S%s" % _seed, "IND_NOGNN", "IND-NOGNN-PUBMEDBERT", _seed,
             "IND-NoGNN+PubMedBERT disjoint-select train_seed=%s" % _seed)


def _names_starting(*prefixes):
    out = []
    for prefix in prefixes:
        out += [n for n in JOBS if n.startswith(prefix)]
    return out


# ---- 2026-10-29 17:20 最终分组（见 REVIEW_V2_RUNBOOK.md §4）：只有 SUBMIT / SKIP 两组 ----
SUBMIT_JOBS = [
    # 近循环：SOC 名称 / 身份 / 结构控制（3 种子）
    "E1B_S42", "E1B_S43", "E1B_S44",
    "SOCDROP_S42", "SOCDROP_S43", "SOCDROP_S44",
    "E1A_S42", "E1A_S43", "E1A_S44",
    # 多种子（seed 42 与 C7 seed 43 取自第一轮，合起来 n=5）
    "V1_S43", "V1_S44", "V1_S45", "V1_S46",
    "V2_S43", "V2_S44", "V2_S45", "V2_S46",
    "C7_S44", "C7_S45", "C7_S46",
    "V3_S43", "V3_S44", "V3_S45", "V3_S46",
    "V4_S43", "V4_S44", "V4_S45", "V4_S46",
    # matched baselines（2026-10-29 追加：降低 no-graph / matched-GNN 单次观察的审查风险）
    "NGPB_S43", "NGPB_S44",
    "GAT_S43", "GAT_S44",
    # 归纳式（disjoint 模型选择）：KEEP n=5、MASK 与 NoGNN 各 1（同协议重算表 5）
    "IND_KEEP_S42", "IND_KEEP_S43", "IND_KEEP_S44", "IND_KEEP_S45", "IND_KEEP_S46",
    "IND_MASK_S42", "IND_NOGNN_S42",
]
_missing_submit = [j for j in SUBMIT_JOBS if j not in JOBS]
if _missing_submit:
    raise SystemExit("SUBMIT 引用了未定义的任务：%s" % _missing_submit)

SHORTCUTS = OrderedDict([
    ("SUBMIT", list(SUBMIT_JOBS)),
    ("SKIP", [j for j in JOBS if j not in SUBMIT_JOBS]),
    ("ALL", list(JOBS.keys())),
])

# 已停用的旧短名 → 说明（直接报错并给出新用法，避免旧命令静默跑成别的集合）
RETIRED_SHORTCUTS = {
    k: "已于 2026-10-29 合并为两组：需要跑的用 SUBMIT（39 个），不需要跑的在 SKIP（61 个）"
    for k in ("G1", "G2", "G3", "MIN", "REC", "CONDE1A", "CONDIND", "COND_E1A", "COND_IND",
              "P0", "P1", "SENS", "SOCCTRL", "IND")
}


# 已移除的任务名 → 原因（防止旧命令/旧笔记里的任务名报"未知任务"时让人困惑）。
REMOVED_JOBS = {
    "SOCEDGE": "与 SOCDROP 计算等价（删 SOC 全部边后 SOC 特征无法到达任何非 SOC 节点）",
    "HARD100": "hard_negative_weight=1.0 与 C4（去硬负样本）数学等价",
}


def resolve_jobs(selection):
    if isinstance(selection, (list, tuple)):
        selection = ",".join(selection)
    parts = [p for chunk in str(selection).split(",") for p in chunk.split()]
    resolved = []
    valid = set(JOBS) | set(SHORTCUTS)
    for p in parts:
        if not p:
            continue
        raw = p.strip().upper()
        key = raw if raw in valid else raw.replace("-", "").replace("_", "")
        if key not in valid:
            if raw in RETIRED_SHORTCUTS or key in RETIRED_SHORTCUTS:
                raise SystemExit("短名 %r %s。" % (p, RETIRED_SHORTCUTS.get(raw) or RETIRED_SHORTCUTS[key]))
            for _prefix, _why in REMOVED_JOBS.items():
                if key.startswith(_prefix):
                    raise SystemExit("任务 %r 已于 2026-10-29 移除：%s。" % (p, _why))
            raise SystemExit("未知任务/短名：%r（可用：%s）" % (p, ", ".join(sorted(valid))))
        for name in SHORTCUTS.get(key, [key]):
            if name not in resolved:
                resolved.append(name)
    if not resolved:
        raise SystemExit("没有选择任何任务；请设置底部 SELECTED_JOBS 或命令行传任务名。")
    return resolved


def _pid_alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return True


def _has_final_ckpt(run_dir):
    if not os.path.isdir(run_dir):
        return False
    return any(name.startswith("best_checkpoint_") and name.endswith(".pt")
               for name in os.listdir(run_dir))


def _job_state(run_dir):
    lock = os.path.join(run_dir, ".run.lock")
    if os.path.exists(lock):
        pid = 0
        try:
            with open(lock, encoding="utf-8") as f:
                pid = int(f.readline().strip() or "0")
        except Exception:
            pid = 0
        if pid > 0 and _pid_alive(pid):
            return "running"
    if _has_final_ckpt(run_dir):
        return "complete"
    if os.path.isdir(run_dir) and any(name.startswith("resume_") and name.endswith(".pt")
                                      for name in os.listdir(run_dir)):
        return "incomplete"
    return "new"


def _cache_is_warm(cache_dir):
    if not cache_dir or not os.path.isdir(cache_dir):
        return False
    names = os.listdir(cache_dir)
    # 只以"图缓存已存在"作为就绪判据。特征缓存文件名随模型/特征类型而变：
    #   SAGE/GAT = `_pubmedbert.pt`（GAT 运行时把 FEATURE_TAG 覆盖回 feature_type），noGNN = `_*_features.pt`；
    #   V3/V4 的随机特征按 train_seed 现场生成、不落盘缓存。
    # 早先额外要求 `_pubmedbert.pt`，导致首个抢到缓存锁的是 V3/V4/NGPB/NGSB/NGCODER
    # 这类不产 pubmedbert 的任务时，其余窗口会空等到它整批跑完（约 11 h），退化为近似串行。
    return any(n.endswith("_graph.pt") for n in names)


def _cache_lock_path(cache_dir):
    return os.path.join(cache_dir, ".cache_build.lock")


def _cache_lock_alive(cache_dir):
    try:
        with open(_cache_lock_path(cache_dir), encoding="utf-8") as f:
            pid = int(f.readline().strip() or "0")
    except Exception:
        return False
    return _pid_alive(pid)


def _try_acquire_cache_lock(cache_dir):
    path = _cache_lock_path(cache_dir)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        pass
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            if not _cache_lock_alive(cache_dir):
                try:
                    os.remove(path)
                    continue
                except OSError:
                    pass
            return False
        except OSError:
            return False
    return False


def _release_cache_lock(cache_dir):
    path = _cache_lock_path(cache_dir)
    try:
        with open(path, encoding="utf-8") as f:
            if f.readline().strip() != str(os.getpid()):
                return
        os.remove(path)
    except OSError:
        pass


def _prepare(job, run_root):
    spec = JOBS[job]
    trainer = TRAINERS[spec["model"]]
    _verify_review_controls(trainer, spec)
    run_dir = os.path.abspath(os.path.join(run_root, job))
    os.makedirs(run_dir, exist_ok=True)
    dst = os.path.join(run_dir, os.path.basename(trainer))
    if os.path.abspath(dst) != os.path.abspath(trainer):
        shutil.copy2(trainer, dst)
    return run_dir, dst


def _make_env(job, run_dir, split_index_file, cache_dir, csv_path, text_model_dir):
    spec = JOBS[job]
    env = os.environ.copy()
    # 清除可能从父 shell 继承的审稿控制变量，避免上一条命令的 SOC 控制污染下一条任务。
    for key in ("MEDDRA_SOC_NAME_PERMUTE", "MEDDRA_SOC_ZERO",
                "MEDDRA_SOC_DROP", "MEDDRA_SOC_EDGE_MASK",
                "MEDDRA_LINK_SPLIT_FILE", "MEDDRA_TAU", "MEDDRA_CL_W",
                "MEDDRA_SIB_W", "MEDDRA_SIB2_W", "MEDDRA_PARENT_W",
                "MEDDRA_HARD_W", "MEDDRA_TRAIN_SEED", "MEDDRA_SPLIT_SEED",
                "MEDDRA_RUN_DIR", "MEDDRA_EXPERIMENT", "MEDDRA_RESUME_FILE",
                "MEDDRA_IND_SELECT", "MEDDRA_KEEP_EDGES"):
        env.pop(key, None)
    env["MEDDRA_RUN_DIR"] = run_dir
    env["MEDDRA_EXPERIMENT"] = spec["experiment"]
    env["MEDDRA_SPLIT_SEED"] = "42"     # 与官方 split_indices.json 的 split_seed 对齐
    env["MEDDRA_TRAIN_SEED"] = spec["seed"]
    env["PYTHONIOENCODING"] = "utf-8"
    if split_index_file:
        env["MEDDRA_SPLIT_INDEX_FILE"] = os.path.abspath(split_index_file)
    if cache_dir:
        env["MEDDRA_CACHE_DIR"] = os.path.abspath(cache_dir)
    if csv_path:
        env["MEDDRA_CSV_PATH"] = os.path.abspath(csv_path)
    if text_model_dir:
        env["MEDDRA_TEXT_MODEL_DIR"] = os.path.abspath(text_model_dir)
    for k, v in spec["env"].items():
        env[k] = v
    return env


def _launch(job, run_root, split_index_file, cache_dir, csv_path, text_model_dir):
    run_dir, dst = _prepare(job, run_root)
    env = _make_env(job, run_dir, split_index_file, cache_dir, csv_path, text_model_dir)
    proc = subprocess.Popen([sys.executable, dst], cwd=run_dir, env=env)
    _RUNNING[job] = proc
    return proc, run_dir


def _terminate_all():
    for proc in list(_RUNNING.values()):
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


def _print_plan(run_root, resolved, split_index_file, cache_dir, csv_path, text_model_dir, dry_run):
    print("=" * 78)
    print("JBI review experiments - plan (%d jobs)" % len(resolved))
    _used = sorted({JOBS[j]["model"] for j in resolved})
    print("trainers: %s" % " / ".join(_used))
    print("run_root: %s" % os.path.abspath(run_root))
    print("split   : %s" % (split_index_file or "UNSET -> will regenerate a split (NOT recommended)"))
    print("cache   : %s" % (cache_dir or "UNSET -> each job builds its own cache"))
    print("csv     : %s" % (csv_path or "UNSET -> trainer default / auto-locate"))
    print("text    : %s" % (text_model_dir or "UNSET -> trainer default"))
    print("dry_run : %s" % dry_run)
    print("-" * 78)
    for job in resolved:
        spec = JOBS[job]
        extra = " ".join("%s=%s" % (k, v) for k, v in spec["env"].items()) or "-"
        print("%-14s exp=%-3s seed=%-2s  %-34s  [%s]" % (
            job, spec["experiment"], spec["seed"], spec["desc"], extra))
    print("=" * 78)
    print("估算：%d 任务 * 约 11 GPU 小时/任务 = 约 %d GPU 小时（单卡）"
          % (len(resolved), len(resolved) * 11))


def _ps_quote(arg):
    arg = str(arg)
    return "'" + arg.replace("'", "''") + "'"


def _print_window_commands(resolved, run_root, split_index_file, cache_dir, csv_path, text_model_dir):
    script = os.path.abspath(__file__)
    print("=" * 78)
    print("每个实验一个窗口：复制下面的命令，在各自的命令行窗口中运行")
    print("-" * 78)
    for job in resolved:
        spec = JOBS[job]
        # SapBERT/CODER 在内部会清空 MEDDRA_TEXT_MODEL_DIR；打印命令时也省略该参数，
        # 避免把 --text-model-dir /data/hf_models/pubmedbert 复制进 CODER/SapBERT 窗口造成误读。
        job_text_model_dir = text_model_dir
        if spec["env"].get("MEDDRA_TEXT_MODEL_DIR") == "":
            job_text_model_dir = ""
        parts = [_ps_quote(sys.executable), _ps_quote(script), job, "--run", "--jobs", "1"]
        for flag, val in (("--run-root", run_root),
                          ("--split-index-file", split_index_file),
                          ("--cache-dir", cache_dir),
                          ("--csv-path", csv_path),
                          ("--text-model-dir", job_text_model_dir)):
            if val:
                parts += [flag, _ps_quote(val)]
        print(" ".join(parts))
    print("-" * 78)
    print("重点：把所有窗口的命令同时启动；第一个窗口会负责建共享缓存，其余窗口自动等待缓存就绪后开始。")


def _run_serial(resolved, run_root, split_index_file, cache_dir, csv_path, text_model_dir):
    fails = []
    for i, job in enumerate(resolved, 1):
        print("[%02d/%02d] start %s" % (i, len(resolved), job))
        proc, _ = _launch(job, run_root, split_index_file, cache_dir, csv_path, text_model_dir)
        rc = proc.wait()
        _RUNNING.pop(job, None)
        if rc == 0:
            print("        ok   %s" % job)
        else:
            print("        FAIL %s (rc=%d)" % (job, rc))
            fails.append(job)
    return fails


def _run_parallel(resolved, run_root, split_index_file, cache_dir, csv_path, text_model_dir, jobs):
    pending = list(resolved)
    running = {}
    fails = []
    started = 0
    while pending or running:
        while pending and len(running) < jobs:
            job = pending.pop(0)
            proc, _ = _launch(job, run_root, split_index_file, cache_dir, csv_path, text_model_dir)
            running[job] = proc
            started += 1
            print("[start %02d/%02d] %s" % (started, len(resolved), job))
        done = []
        for job, proc in running.items():
            if proc.poll() is not None:
                done.append(job)
                _RUNNING.pop(job, None)
                if proc.returncode == 0:
                    print("[ ok  ] %s" % job)
                else:
                    print("[FAIL ] %s (rc=%d)" % (job, proc.returncode))
                    fails.append(job)
        for job in done:
            del running[job]
        if running:
            time.sleep(5)
    return fails


def _run_launch_list(launch, run_root, split_index_file, cache_dir, csv_path, text_model_dir, parallel):
    if len(launch) == 1 or parallel <= 1:
        return _run_serial(launch, run_root, split_index_file, cache_dir, csv_path, text_model_dir)
    return _run_parallel(launch, run_root, split_index_file, cache_dir, csv_path, text_model_dir, parallel)


def main(jobs_cli, run_root, split_index_file, cache_dir, csv_path, text_model_dir,
         dry_run, parallel, skip_e1b, rerun, selection_default, print_commands=False):
    resolved = resolve_jobs(jobs_cli or selection_default)
    if skip_e1b:
        resolved = [j for j in resolved if not j.startswith("E1B_")]

    # 只检查本次选中任务用到的训练器（只跑 SAGE 时不要求归纳式文件存在）
    for _model in sorted({JOBS[j]["model"] for j in resolved}):
        trainer = TRAINERS[_model]
        if not os.path.exists(trainer):
            raise SystemExit("找不到训练脚本：%s" % trainer)
        _check_trainer_file(trainer)
        for _j in resolved:
            if JOBS[_j]["model"] == _model:
                _verify_review_controls(trainer, JOBS[_j])

    _print_plan(run_root, resolved, split_index_file, cache_dir, csv_path, text_model_dir, dry_run)
    if print_commands:
        _print_window_commands(resolved, run_root, split_index_file, cache_dir, csv_path, text_model_dir)
        return 0
    if dry_run:
        print("--dry-run：仅打印计划，未实际运行。（正式运行：底部 DRY_RUN=False 或加 --run）")
        return 0

    if not split_index_file:
        print("警告：未设置 split_indices.json，各任务可能重新划分，无法与主表同口径比较。")
    if not cache_dir:
        print("提醒：未设置共享缓存目录，每个任务都会重新编码 PubMedBERT，耗时极大。")
    elif not _cache_is_warm(cache_dir):
        print("提醒：共享缓存尚不完整；多窗口并发时会有一个窗口先建缓存，其余窗口等待就绪后启动。")

    launch = []
    for job in resolved:
        st = _job_state(os.path.join(run_root, job))
        if st == "running":
            print("[skip] %s 正在运行（.run.lock 存活），跳过以避免双写。" % job)
            continue
        if st == "complete" and not rerun:
            print("[skip] %s 已完成（best_checkpoint_* 已存在）；加 --rerun 可强制重跑。" % job)
            continue
        launch.append(job)

    if not launch:
        print("没有需要运行的任务。")
        return 0

    fails = []
    total_launch = len(launch)
    cache_owner = False
    try:
        # 多个命令行窗口共享同一缓存目录时，只允许一个窗口先建缓存（图 + PubMedBERT 特征），
        # 其余窗口等待缓存就绪后再启动各自实验，避免并发双写同一个缓存文件。
        if cache_dir and not _cache_is_warm(cache_dir):
            cache_owner = _try_acquire_cache_lock(cache_dir)
            waited = 0
            while not cache_owner and not _cache_is_warm(cache_dir):
                time.sleep(10)
                waited += 10
                if waited > CACHE_WAIT_TIMEOUT:
                    print("[cache] 等待共享缓存超时；继续启动本窗口任务（可能自行建缓存并发生并发写）。")
                    break
                cache_owner = _try_acquire_cache_lock(cache_dir)
            if cache_owner:
                print("[cache] 本窗口持有缓存构建锁，先跑本窗口任务并负责建成共享缓存。")
            elif _cache_is_warm(cache_dir):
                print("[cache] 共享缓存已由其他窗口建成，直接启动本窗口任务。")

        if cache_owner:
            try:
                if len(launch) > 1 and parallel > 1 and cache_dir and not _cache_is_warm(cache_dir):
                    warm_job = launch[0]
                    print("[warm] 本窗口先串行运行 %s 建立共享缓存..." % warm_job)
                    proc, _ = _launch(warm_job, run_root, split_index_file, cache_dir, csv_path, text_model_dir)
                    rc = proc.wait()
                    _RUNNING.pop(warm_job, None)
                    if rc == 0:
                        print("        warm ok   %s" % warm_job)
                    else:
                        print("        warm FAIL %s (rc=%d)" % (warm_job, rc))
                        fails.append(warm_job)
                    launch = launch[1:]
                fails += _run_launch_list(launch, run_root, split_index_file, cache_dir,
                                          csv_path, text_model_dir, parallel)
            finally:
                _release_cache_lock(cache_dir)
        else:
            fails += _run_launch_list(launch, run_root, split_index_file, cache_dir,
                                      csv_path, text_model_dir, parallel)
    except KeyboardInterrupt:
        print("\n收到中断，正在终止已启动的子进程...")
        _terminate_all()
        print("已终止。未完成任务的断点（resume_*.pt）会保留，重跑本脚本即可续训。")
        return 2

    print("=" * 78)
    if not fails:
        print("全部 %d 个任务完成。" % total_launch)
        return 0
    print("失败任务：%s" % ", ".join(fails))
    return 1


# ============================================================
#  ★ 运行配置（只需改这一块）★
# ============================================================
# 新实验输出根目录（只写新结果，不覆盖既有 SAGE/GAT/NOGNN 目录）。
# 服务器优先：MEDDRA_REVIEW_ROOT > MEDDRA_OUTPUT_ROOT 同级 *_review > 本地 Windows 默认。
def _default_review_root():
    _rv = os.environ.get("MEDDRA_REVIEW_ROOT")
    if _rv:
        return _rv
    _out = os.environ.get("MEDDRA_OUTPUT_ROOT")
    if _out:
        _out = _out.replace(os.sep, "/").rstrip("/")
        _parent = os.path.dirname(_out)
        _base = os.path.basename(_out)
        return os.path.join(_parent, _base + "_review") if _parent else (_base + "_review")
    return r"D:\MedDRA模型文件\JBI\REVIEW"

RUN_ROOT = _default_review_root()

# 官方三分划分索引（必须与主表同口径；split_seed=42）。
SPLIT_INDEX_FILE = os.environ.get("MEDDRA_SPLIT_INDEX_FILE") or r"D:\MedDRA模型文件\JBI\split_indices.json"

# 共享缓存 / 数据 / 文本模型（优先读环境变量；留空则用训练脚本默认）。
# TEXT_MODEL_DIR = 本地 PubMedBERT 权重目录；服务器用 /data/hf_models/pubmedbert。
CACHE_DIR = os.environ.get("MEDDRA_CACHE_DIR") or ""
CSV_PATH = os.environ.get("MEDDRA_CSV_PATH") or r"D:\MedDRA模型文件\JBI\MEDDRA_DATA_29_0.csv"
TEXT_MODEL_DIR = os.environ.get("MEDDRA_TEXT_MODEL_DIR") or ""

# 要跑哪些任务：SUBMIT（投稿需要的 39 个）；SKIP / ALL 仅备查。
SELECTED_JOBS = "SUBMIT"

# True = 先看计划不真跑；确认后改成 False（或命令行加 --run）。
DRY_RUN = True

# 并发数：单窗口单任务模式传 --jobs 1；只有用一个窗口跑整组时才用多并发。
PARALLEL = 8
# ============================================================


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="JBI review experiments (SAGE/NOGNN/GAT/IND_SAGE/IND_NOGNN)")
    ap.add_argument("jobs", nargs="*", help="任务名或短名 SUBMIT / SKIP / ALL；留空用底部 SELECTED_JOBS")
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--split-index-file", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--text-model-dir", default=None)
    ap.add_argument("--jobs", type=int, default=None, dest="parallel")
    ap.add_argument("--skip-e1b", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="强制重跑已完成的目录")
    ap.add_argument("--print-commands", action="store_true", help="只打印每个实验一个窗口的 PowerShell 启动命令")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")

    a = ap.parse_args()
    _dry = True if a.dry_run else (False if a.run else DRY_RUN)
    _parallel = PARALLEL if a.parallel is None else a.parallel
    sys.exit(main(
        jobs_cli=a.jobs,
        run_root=a.run_root or RUN_ROOT,
        split_index_file=a.split_index_file or SPLIT_INDEX_FILE,
        cache_dir=a.cache_dir or CACHE_DIR,
        csv_path=a.csv_path or CSV_PATH,
        text_model_dir=a.text_model_dir or TEXT_MODEL_DIR,
        dry_run=_dry,
        parallel=_parallel,
        skip_e1b=a.skip_e1b,
        rerun=a.rerun,
        selection_default=SELECTED_JOBS,
        print_commands=a.print_commands,
    ))
