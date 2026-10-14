"""A12 · 仪器分量不变性的**操作点扫描**（把 §3.6 的 n=2 升级为一条实测曲线）

起因（2026-07-25 预投稿评审，投稿前第 2 项）：
  §3.6 现在只有两个模型工作点（弱 R²≈0.393、强 R²≈0.819），据此声称
  「仪器相关的绝对 RMSEP 分量在模型变好时基本不变，因而它占 MSEP 的比例上升」。
  评审判定：**n=2 是审稿人最容易下笔的地方**——两点连一条线，既无法证伪也无法定量。
  要求扫 6–10 个工作点覆盖 R²≈0.4–0.85，画「仪器相关 RMSEP 对总 RMSEP」：
  平 ⇒ 不变性由两点升级为实测曲线；有斜率 ⇒ 如实报斜率并削弱结论措辞。
  **两种结果都能用**——这正是它可证伪的地方。

预注册式纪律（本脚本的红线）：
  · 配置清单 CONFIGS 在**看到任何 R² 之前**写死，见下方常量；
  · **全部配置一律报告**，不得因 R² 落点不好看而事后增删；
  · 不做任何以 R² 为目标的挑选（那正是 §4.3 限制(6) 已披露的那类错误）。

口径与 §3.6 的关系：
  97instrbudget 用 sklearn 的 GroupKFold（确定性、无种子维度），因此那两个点没有区间。
  本脚本要报斜率，就必须有不确定度，故改用**按果随机分配到 5 折**的分组 CV，
  并用 Formal 5 固定种子重复；区间一律用 §2.3 声明的**以种子为簇**的两级自助。
  这与 GroupKFold 不是同一分折规则，故本扫描自成一套口径：
  **§3.6 的两个配置也在本口径下重跑**（配置 W 与 S），以便曲线内部自洽、
  并可看出它们落在曲线的哪个位置。正文引用时不得与 0.393/0.819 混用。

仪器分量定义（与 97instrbudget 表d 逐字一致，不另起炉灶）：
  对每个果，取其在不同设备上的 out-of-fold 预测的样本 SD；
  var_instr = mean(SD²)；「换台仪器就会变的那部分 RMSEP」= sqrt(var_instr)；
  占比 = var_instr / RMSEP²。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

import os as _os0
_ROOT = _os0.environ.get("HSI_DATA_ROOT",
                        _os0.path.join(_os0.path.dirname(_os0.path.abspath(__file__)), "..", "data"))
DATA = _os0.path.join(_ROOT, "012_kiwifruit_nir_drymatter", "kiwifruit_dat.csv")
SEEDS = [20060515, 20041210, 19810915, 2023, 2024]        # Formal 5，与全稿一致
N_FOLD, N_INNER, N_BOOT = 5, 4, 2000

# ══ 配置清单：在看到任何结果之前写死 ══════════════════════════════════════════
# (标签, 预处理, 最低波长 nm, 成分数)  成分数 None = 训练折内分组 CV 选
CONFIGS = [
    ("C1  raw 全谱 · 2 成分",            "raw",  402,  2),
    ("C2  raw 全谱 · 6 成分",            "raw",  402,  6),
    ("W   SNV 全谱 · 10 成分（=§3.6 弱）", "snv",  402, 10),
    ("C3  SNV 全谱 · 4 成分",            "snv",  402,  4),
    ("C4  SNV ≥700nm · 10 成分",         "snv",  700, 10),
    ("C5  SG1 全谱 · 6 成分",            "sg1",  402,  6),
    ("C6  SG1 ≥700nm · 6 成分",          "sg1",  700,  6),
    ("C7  SG2 ≥700nm · 10 成分",         "sg2",  700, 10),
    ("S   SG1 ≥700nm · CV 选（=§3.6 强）", "sg1",  700, None),
    ("C8  SNV+SG1 ≥800nm · CV 选",       "snv_sg1", 800, None),
]


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA, low_memory=False)
    wave = sorted([c for c in df.columns if c.startswith("X") and c[1:].isdigit()],
                  key=lambda c: int(c[1:]))
    df = df.dropna(subset=["sample_id", "device"])
    bad = [c for c in wave if df[c].isna().any()]      # 五台设备公共可用区间
    return df, [c for c in wave if c not in bad]


def snv(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=1, keepdims=True)) / sd


def preprocess(Xraw: np.ndarray, wl: np.ndarray, kind: str, lo: float) -> np.ndarray:
    X = Xraw[:, wl >= lo]
    if kind == "raw":
        return X
    if kind == "snv":
        return snv(X)
    if kind == "sg1":
        return savgol_filter(X, 11, 2, deriv=1, axis=1)
    if kind == "sg2":
        return savgol_filter(X, 11, 2, deriv=2, axis=1)
    if kind == "snv_sg1":
        return savgol_filter(snv(X), 11, 2, deriv=1, axis=1)
    raise ValueError(kind)


def grouped_folds(groups: np.ndarray, seed: int, k: int = N_FOLD) -> list[np.ndarray]:
    """按**果**随机等分到 k 折（同一果整体进同一折），返回每折的布尔掩膜。"""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    assign = {g: i % k for i, g in enumerate(rng.permutation(uq))}
    fold = np.array([assign[g] for g in groups])
    return [fold == i for i in range(k)]


def run_config(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_comp: int | None, seed: int) -> dict:
    """一个 (配置, 种子) 的分组 5 折 CV；SST 一律以**训练折均值**为基准（M9 口径）。"""
    pred = np.full(len(y), np.nan)
    sst = 0.0
    for te in grouped_folds(groups, seed):
        tr = ~te
        if n_comp is None:                              # 训练折内再按果分组选成分数
            best, berr = 2, np.inf
            inner = grouped_folds(groups[tr], seed, N_INNER)
            for c in range(2, min(25, X[tr].shape[1]) + 1, 2):
                errs = []
                for ib in inner:
                    ia = ~ib
                    if ia.sum() < c + 2 or ib.sum() == 0:
                        continue
                    m = PLSRegression(c).fit(X[tr][ia], y[tr][ia])
                    errs.append(np.mean((m.predict(X[tr][ib]).ravel() - y[tr][ib]) ** 2))
                if errs and np.mean(errs) < berr - 1e-9:
                    berr, best = float(np.mean(errs)), c
        else:
            best = min(n_comp, X.shape[1])
        pred[te] = PLSRegression(best).fit(X[tr], y[tr]).predict(X[te]).ravel()
        sst += float(np.sum((y[te] - y[tr].mean()) ** 2))

    sse = float(np.sum((pred - y) ** 2))
    rmsep = float(np.sqrt(np.mean((pred - y) ** 2)))
    s = pd.Series(pred, index=groups).groupby(level=0)
    sd_within = s.std(ddof=1)                            # 同果跨设备的预测离散
    var_instr = float(np.nanmean(sd_within ** 2))
    return {"R2": 1 - sse / sst, "RMSEP": rmsep,
            "instr_rmsep": float(np.sqrt(var_instr)),
            "instr_share": var_instr / rmsep ** 2}


def boot_seed_cluster(per_seed: dict[int, list[float]], stat, seed: int = 0) -> tuple[float, float]:
    """以种子为簇的两级自助（§2.3 全稿统一口径）。stat 作用于重抽后的合并样本。"""
    rng = np.random.default_rng(seed)
    keys = list(per_seed)
    out = []
    for _ in range(N_BOOT):
        picked = [np.asarray(per_seed[keys[i]], float) for i in rng.integers(0, len(keys), len(keys))]
        vals = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in picked])
        out.append(stat(vals))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    log = Logger(os.path.splitext(os.path.basename(__file__))[0])
    sys.stdout = log
    df, wave = load()
    multi = df.groupby("sample_id")["device"].nunique()
    sub = df[df["sample_id"].isin(multi[multi >= 2].index)].dropna(subset=["DM"]).copy()
    Xraw = sub[wave].to_numpy(dtype=float)
    wl = np.array([float(c[1:]) for c in wave])
    y = sub["DM"].to_numpy(dtype=float)
    groups = sub["sample_id"].to_numpy()
    log.log(f"[A12] 有标签且 ≥2 台设备扫过：{len(sub)} 条谱 / {len(np.unique(groups))} 果 / "
            f"{sub['device'].nunique()} 台设备；波段 {len(wave)}")
    log.log(f"[A12] 事先写死 {len(CONFIGS)} 个配置 × {len(SEEDS)} 种子，全部报告，无事后增删。\n")

    rows, per_cfg = [], {}
    for label, kind, lo, nc in CONFIGS:
        X = preprocess(Xraw, wl, kind, lo)
        res = [run_config(X, y, groups, nc, s) for s in SEEDS]
        per_cfg[label] = res
        m = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        sd = {k: float(np.std([r[k] for r in res], ddof=1)) for k in res[0]}
        rows.append({
            "配置": label, "预处理": kind, "起始波长nm": lo,
            "成分数": "CV 选" if nc is None else nc, "波段数": X.shape[1],
            "R²（5 种子均值）": m["R2"], "R² 跨种子SD": sd["R2"],
            "RMSEP（%DM）": m["RMSEP"], "RMSEP 跨种子SD": sd["RMSEP"],
            "仪器相关 RMSEP 分量（%DM）": m["instr_rmsep"],
            "仪器分量 跨种子SD": sd["instr_rmsep"],
            "仪器分量占 MSEP 比例": m["instr_share"],
        })
        log.log(f"  {label:32s} R²={m['R2']:+.4f}  RMSEP={m['RMSEP']:.4f}  "
                f"仪器分量={m['instr_rmsep']:.4f}  占比={m['instr_share']:.3f}")
    tab_a = pd.DataFrame(rows).sort_values("R²（5 种子均值）").reset_index(drop=True)

    # ── 表b：斜率检验（核心可证伪判据）──────────────────────────────────────
    # 问：仪器相关的**绝对**分量是否随 R² 变化？零假设「不变」⇔ 斜率 = 0。
    # 逐种子拟合 instr_rmsep ~ a + b·R²，再对 5 个斜率做以种子为簇的自助。
    slopes = {}
    for s_i, s in enumerate(SEEDS):
        xs = np.array([per_cfg[l][s_i]["R2"] for l, *_ in CONFIGS])
        ys = np.array([per_cfg[l][s_i]["instr_rmsep"] for l, *_ in CONFIGS])
        slopes[s] = [float(np.polyfit(xs, ys, 1)[0])]
    b_mean = float(np.mean([v[0] for v in slopes.values()]))
    b_lo, b_hi = boot_seed_cluster(slopes, np.mean)
    # 对照：占比对 R² 的斜率（理论预期为**正**——分子不变、分母缩小）
    sh_slopes = {}
    for s_i, s in enumerate(SEEDS):
        xs = np.array([per_cfg[l][s_i]["R2"] for l, *_ in CONFIGS])
        ys = np.array([per_cfg[l][s_i]["instr_share"] for l, *_ in CONFIGS])
        sh_slopes[s] = [float(np.polyfit(xs, ys, 1)[0])]
    sh_mean = float(np.mean([v[0] for v in sh_slopes.values()]))
    sh_lo, sh_hi = boot_seed_cluster(sh_slopes, np.mean)

    r2s = tab_a["R²（5 种子均值）"].to_numpy()
    inst = tab_a["仪器相关 RMSEP 分量（%DM）"].to_numpy()
    # 拟合诊断——2026-07-26 claim audit C033/C003 要求落盘：
    # 正文同时引用「实测极差」与「拟合趋势」，二者数量级差 20 倍，必须分别存、分别命名，
    # 绝不可用拟合趋势去描述实测变动（审计判该混用为 critical）。
    _b1, _b0 = np.polyfit(r2s, inst, 1)
    _fit = _b0 + _b1 * r2s
    _ss_res = float(((inst - _fit) ** 2).sum())
    _ss_tot = float(((inst - inst.mean()) ** 2).sum())
    _ols_r2 = 1.0 - _ss_res / _ss_tot
    _span = float(r2s.max() - r2s.min())
    _fit_change = float(_b1 * _span)
    _fit_pct = 100.0 * _fit_change / float(inst.mean())
    _resid_sd = float(np.sqrt(_ss_res / (len(r2s) - 2)))
    _obs_pct = 100.0 * float(inst.max() - inst.min()) / float(inst.mean())
    tab_b = pd.DataFrame([
        {"量": "工作点个数", "值": len(CONFIGS), "CI95 下限": None, "CI95 上限": None,
         "读法": "预注册清单，全部报告"},
        {"量": "R² 覆盖范围", "值": f"{r2s.min():.3f} – {r2s.max():.3f}",
         "CI95 下限": None, "CI95 上限": None, "读法": "面板要求覆盖 0.4–0.85"},
        # 2026-07-27 claim audit T4：正文写了「每种子把果随机等分到 5 折」，但这个 5
        # 此前只存在于本脚本的 N_FOLD 常量里，任何工作簿都查不到 → 判 mismatch。
        # 本项目规矩④：写进论文的量必须由脚本算出并落盘。分折规则也算，故补落盘。
        {"量": "外层分折数 N_FOLD（本扫描自己的分折规则）", "值": N_FOLD,
         "CI95 下限": None, "CI95 上限": None,
         "读法": "每个种子把**果**随机等分到这么多折；与确定性分组 k 折是两套规则，数字不可互换"},
        {"量": "内层选成分数的折数 N_INNER", "值": N_INNER,
         "CI95 下限": None, "CI95 上限": None, "读法": "在训练折内按果分组再切，用于选 PLS 成分数"},
        {"量": "固定种子表", "值": ", ".join(str(s) for s in SEEDS),
         "CI95 下限": None, "CI95 上限": None,
         "读法": f"共 {len(SEEDS)} 个，与全稿一致；表a 各值为这些种子的均值"},
        {"量": "自助重抽次数 N_BOOT", "值": N_BOOT,
         "CI95 下限": None, "CI95 上限": None, "读法": "以种子为簇的两级自助，用于斜率 CI95"},
        {"量": "仪器相关 RMSEP 分量 · 极差", "值": float(inst.max() - inst.min()),
         "CI95 下限": None, "CI95 上限": None,
         "读法": "绝对分量在全部工作点上的最大变动"},
        {"量": "仪器相关 RMSEP 分量 · **实测**相对极差(%)", "值": _obs_pct,
         "CI95 下限": None, "CI95 上限": None,
         "读法": "十点中最大与最小之差 ÷ 均值。**这是实测变动**，含配置间散布，"
                 "不可与下面的「拟合趋势」混用"},
        {"量": "回归解释的方差比例 OLS R²", "值": _ols_r2,
         "CI95 下限": None, "CI95 上限": None,
         "读法": "R² 只解释了该分量跨配置变动的这一小部分"},
        {"量": "**拟合趋势**在全 R² 区间上的变化量（%DM）", "值": _fit_change,
         "CI95 下限": None, "CI95 上限": None,
         "读法": "斜率 × R² 跨度。这是**趋势**不是实测极差"},
        {"量": "**拟合趋势**变化量 ÷ 分量均值(%)", "值": _fit_pct,
         "CI95 下限": None, "CI95 上限": None,
         "读法": "正文引用此值时必须写明「拟合趋势」，不得说成分量的实测变动"},
        {"量": "回归残差 SD（%DM）", "值": _resid_sd,
         "CI95 下限": None, "CI95 上限": None,
         "读法": "与模型好坏无关的配置间散布"},
        {"量": "**斜率 d(仪器分量)/d(R²)**", "值": b_mean, "CI95 下限": b_lo, "CI95 上限": b_hi,
         "读法": "区间含 0 ⇒ 不变性成立；不含 0 ⇒ 如实报斜率并削弱结论"},
        {"量": "（对照）斜率 d(占MSEP比例)/d(R²)", "值": sh_mean,
         "CI95 下限": sh_lo, "CI95 上限": sh_hi,
         "读法": "机理预期为正：分子不变而分母随模型变好而缩小"},
    ])

    log.log("\n" + tab_b.to_string(index=False))
    verdict = ("不变性成立（斜率 CI 含 0）" if b_lo <= 0 <= b_hi
               else f"存在斜率（CI 不含 0），须如实报告并削弱结论措辞")
    log.log(f"\n[A12] 判定：{verdict}")

    out = write_script_workbook(__file__, {
        0: ("操作点扫描明细", tab_a),
        "斜率检验与判定": tab_b,
    })
    log.log(f"→ {out}")
    sys.stdout = log._stdout  # noqa: SLF001


if __name__ == "__main__":
    main()
