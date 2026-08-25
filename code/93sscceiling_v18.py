"""C4 pilot · 第 1 段：001 苹果个体内空间异质性与「可达上界」的量化。

背景（v18 / 2026-07-25）
------------------------
数据：05data/001_apple_hyperspectral_multiyear/ 2025 年批次。
课题组负责人已确认测法：**每个苹果 5 个采样面（顶部/底部/侧面×3），每个面由折光仪
在破坏性测量后各自给出一个 SSC 读数**。故面间差异是真实的空间化学异质性，不是同点重
复读数噪声（ATAGO 类折光仪标称精度约 ±0.2 °Brix）。

本脚本回答三件事，全部落 xlsx 供论文直接引用：
  1. 数据质量：定位并分类异常值（物理不可能 vs 生物学可能的低糖组织）。
  2. 方差分解：果间 / 果内(面间) 方差分量、ICC、单面与 m 面均值的标准误；分产地重做。
  3. 可达上界：若模型实际感知的是「整果潜在 SSC」μ_i，而评测用的真值是 m 个面的均值，
     则任何模型的 R² 上界为  ICC_m = σ²_a / (σ²_a + σ²_f/m)，RMSE 下界为 σ_f/√m。
     这条给出「报告的 RMSEP 要配几面真值才可信」的可操作判据。

统计口径
--------
单向随机效应模型   Y_ij = μ + a_i + e_ij ,  a_i ~ (0, σ²_a) ,  e_ij ~ (0, σ²_f) ,  j = 1..k
平衡设计 k=5 时用经典 ANOVA 矩估计：σ̂²_f = MSW ,  σ̂²_a = (MSB − MSW)/k 。
置信区间一律按**果为单位**的 bootstrap（重抽整果，保持果内 5 个面绑定），B=2000，种子 2004。

清洗口径（预注册，先定后跑）
--------------------------
  · 剔除 SSC > 20 °Brix 的单元格 —— 苹果 SSC 生理上限约 18–20，超出即录入错误。
  · **不剔除**低值（<5 °Brix）—— 局部未熟/腐烂组织真能到 3–5 °Brix，那正是本文要刻画的
    现象本身；数据说明文件亦记录了局部腐烂样品。剔除低值等于把结论先验地做小。
  · 主口径 = 整果剔除（某果有任一不可能值则该果 5 面全弃），保守。敏感性分析里给
    「只置该格为缺失」与「另剔低值」两种口径的对照。
"""

from __future__ import annotations

import os
import sys
import zlib
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEED = 2004
N_BOOT = 2000
K_FACES = 5
IMPOSSIBLE_HI = 20.0          # °Brix，苹果生理上限之上 → 录入错误
LOW_FLAG = 5.0                # °Brix，仅作标记与敏感性分析，主口径不剔除
REFRACTOMETER_PRECISION = 0.2  # °Brix，ATAGO 类数显折光仪标称

import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_DIR = os.path.join(DATA_ROOT, "001_apple_hyperspectral_multiyear")
SSC_FILE = "新疆-山东-甘肃苹果糖度数据2025.csv"
SPEC_FILE = "新疆-山东-甘肃苹果光谱数据2025.csv"
FACE_COLS = ["顶部", "底部", "侧面1", "侧面2", "侧面3"]

logger = Logger(__file__)


# --------------------------------------------------------------------------- #
# 方差分解
# --------------------------------------------------------------------------- #
def variance_components(mat: np.ndarray) -> dict[str, float]:
    """平衡单向随机效应 ANOVA 矩估计。mat: (n_fruit, k_faces)，无缺失。"""
    n, k = mat.shape
    row_mu = mat.mean(axis=1)
    grand = mat.mean()
    ms_between = k * ((row_mu - grand) ** 2).sum() / (n - 1)
    ms_within = ((mat - row_mu[:, None]) ** 2).sum() / (n * (k - 1))
    var_apple = max((ms_between - ms_within) / k, 0.0)
    var_face = ms_within
    icc = var_apple / (var_apple + var_face) if (var_apple + var_face) > 0 else np.nan
    return {
        "n_fruit": n,
        "MS_between": ms_between,
        "MS_within": ms_within,
        "var_apple": var_apple,
        "var_face": var_face,
        "ICC": icc,
        "within_share": 1 - icc,
        "sd_face": np.sqrt(var_face),
        "sem_mean5": np.sqrt(var_face / k),
    }


def boot_ci(mat: np.ndarray, stat_key: str, n_boot: int = N_BOOT,
            seed: int = SEED) -> tuple[float, float]:
    """按果重抽的 bootstrap 百分位区间（果内 5 面整体绑定，保持相关结构）。"""
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = variance_components(mat[idx])[stat_key]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def attainable_ceiling(var_apple: float, var_face: float,
                       m_grid: tuple[int, ...] = (1, 2, 3, 4, 5, 10)) -> pd.DataFrame:
    """真值取 m 个面均值时，任何「只能预测整果潜在 SSC」的模型的 R² 上界 / RMSE 下界。"""
    rows = []
    for m in m_grid:
        var_truth = var_apple + var_face / m       # 该真值自身的方差
        r2_max = var_apple / var_truth             # 完美预测 μ_i 时对该真值的 R²
        rmse_min = np.sqrt(var_face / m)           # 完美预测 μ_i 时的不可约 RMSE
        rows.append({
            "m_faces_as_truth": m,
            "var_of_truth": var_truth,
            "R2_upper_bound": r2_max,
            "RMSE_lower_bound_Brix": rmse_min,
            "note": "单面真值（多数文献口径）" if m == 1 else
                    ("本数据集黄金真值" if m == 5 else ""),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    np.random.seed(SEED)
    logger.log(f"[93sscceiling] SEED={SEED} N_BOOT={N_BOOT}")

    ssc = pd.read_csv(os.path.join(DATA_DIR, SSC_FILE), encoding="gbk")
    ssc.columns = ["id"] + FACE_COLS
    ssc["origin"] = ssc["id"].astype(str).str[:2]
    raw = ssc[FACE_COLS].apply(pd.to_numeric, errors="coerce")
    logger.log(f"糖度表 {ssc.shape}；产地 {dict(Counter(ssc['origin']))}")

    # ---------- 表a：数据质量 ----------
    quality_rows = []
    long = raw.stack()
    hi = long[long > IMPOSSIBLE_HI]
    lo = long[long < LOW_FLAG]
    for (i, col), val in hi.items():
        others = raw.loc[i].drop(col).dropna().values
        quality_rows.append({
            "苹果编码": ssc.loc[i, "id"], "采样面": col, "SSC": val,
            "同果其余面": ", ".join(f"{v:.1f}" for v in others),
            "判定": "物理不可能（苹果 SSC 生理上限约 18–20 °Brix）→ 录入错误，剔除",
        })
    for (i, col), val in lo.sort_values().items():
        others = raw.loc[i].drop(col).dropna().values
        quality_rows.append({
            "苹果编码": ssc.loc[i, "id"], "采样面": col, "SSC": val,
            "同果其余面": ", ".join(f"{v:.1f}" for v in others),
            "判定": "生物学可能（局部未熟/腐烂组织）→ 标记但保留",
        })
    tab_quality = pd.DataFrame(quality_rows)

    # 样本量流转
    n_all = len(ssc)
    n_allnan = int(raw.isna().all(axis=1).sum())
    complete_raw = raw.dropna()
    bad_fruit = raw[(raw > IMPOSSIBLE_HI).any(axis=1)].index
    clean = raw.drop(index=bad_fruit).dropna()
    flow = pd.DataFrame([
        {"步骤": "糖度表总行数（= 唯一苹果编码数）", "果数": n_all},
        {"步骤": "扣除 5 面全缺（丢失/腐烂不可用）", "果数": n_all - n_allnan},
        {"步骤": "扣除 5 面不全", "果数": len(complete_raw)},
        {"步骤": f"扣除含 >{IMPOSSIBLE_HI:g} °Brix 录入错误的整果（主口径）", "果数": len(clean)},
    ])

    # ---------- 表b：方差分解 ----------
    vc_rows = []
    vc_all = variance_components(clean.values)
    lo_icc, hi_icc = boot_ci(clean.values, "ICC")
    lo_ws, hi_ws = boot_ci(clean.values, "within_share")
    vc_rows.append({
        "分组": "全体（主口径）", **vc_all,
        "ICC_CI95": f"[{lo_icc:.3f}, {hi_icc:.3f}]",
        "果内占比_CI95": f"[{min(lo_ws, hi_ws):.3f}, {max(lo_ws, hi_ws):.3f}]",
    })
    origins = ssc.loc[clean.index, "origin"]
    for g in sorted(origins.unique()):
        sub = clean[origins == g]
        vc = variance_components(sub.values)
        # crc32 而非内置 hash()：str 的 hash 每个解释器进程都加随机盐，会让 CI 不可复现
        l1, h1 = boot_ci(sub.values, "ICC", seed=SEED + zlib.crc32(g.encode()) % 1000)
        vc_rows.append({"分组": f"{g}（{ {'GS': '甘肃', 'SD': '山东', 'XJ': '新疆'}.get(g, g) }）",
                        **vc, "ICC_CI95": f"[{l1:.3f}, {h1:.3f}]", "果内占比_CI95": ""})
    tab_vc = pd.DataFrame(vc_rows)

    # ---------- 表c：果内异质性幅度 vs 仪器精度 ----------
    sd_w = clean.std(axis=1, ddof=1)
    rng_w = clean.max(axis=1) - clean.min(axis=1)
    het_rows = [
        {"指标": "果内 5 面 SD（中位数）", "值": sd_w.median(), "单位": "°Brix"},
        {"指标": "果内 5 面 SD（均值）", "值": sd_w.mean(), "单位": "°Brix"},
        {"指标": "折光仪标称精度", "值": REFRACTOMETER_PRECISION, "单位": "°Brix"},
        {"指标": "果内 SD 中位数 / 仪器精度", "值": sd_w.median() / REFRACTOMETER_PRECISION,
         "单位": "倍"},
    ]
    for p in (10, 25, 50, 75, 90, 95, 99):
        het_rows.append({"指标": f"果内极差 P{p}", "值": rng_w.quantile(p / 100), "单位": "°Brix"})
    for thr in (1, 2, 3, 5):
        het_rows.append({"指标": f"果内极差 > {thr} °Brix 的果占比",
                         "值": 100 * (rng_w > thr).mean(), "单位": "%"})
    tab_het = pd.DataFrame(het_rows)

    # ---------- 表d：可达上界 ----------
    tab_ceiling = attainable_ceiling(vc_all["var_apple"], vc_all["var_face"])
    for g in sorted(origins.unique()):
        vc = variance_components(clean[origins == g].values)
        sub = attainable_ceiling(vc["var_apple"], vc["var_face"], (1, 5))
        sub.insert(0, "分组", g)
        tab_ceiling = pd.concat(
            [tab_ceiling.assign(**({} if "分组" in tab_ceiling else {"分组": "全体"})), sub],
            ignore_index=True)
    cols = ["分组"] + [c for c in tab_ceiling.columns if c != "分组"]
    tab_ceiling = tab_ceiling[cols]

    # ---------- 表e：采样面位置是否有系统效应 ----------
    centered = clean.sub(clean.mean(axis=1), axis=0)   # 去掉果效应，只看面位置
    face_rows = []
    for col in FACE_COLS:
        d = centered[col]
        se = d.std(ddof=1) / np.sqrt(len(d))
        face_rows.append({
            "采样面": col, "原始均值": clean[col].mean(), "原始SD": clean[col].std(ddof=1),
            "去果效应后偏移": d.mean(), "偏移SE": se, "t": d.mean() / se,
        })
    tab_face = pd.DataFrame(face_rows)
    # 置换检验：面位置标签在果内随机重排，检验偏移幅度
    rng = np.random.default_rng(SEED)
    obs = np.abs(centered.mean(axis=0).values).max()
    perm = np.empty(N_BOOT)
    arr = clean.values
    for b in range(N_BOOT):
        sh = np.apply_along_axis(rng.permutation, 1, arr)
        perm[b] = np.abs((sh - sh.mean(axis=1, keepdims=True)).mean(axis=0)).max()
    p_face = float((perm >= obs).mean())
    tab_face = pd.concat([tab_face, pd.DataFrame([{
        "采样面": "【置换检验】最大|偏移|", "去果效应后偏移": obs,
        "偏移SE": np.nan, "t": np.nan, "原始均值": np.nan, "原始SD": np.nan,
    }, {"采样面": f"【置换检验】p 值（B={N_BOOT}）", "去果效应后偏移": p_face,
        "偏移SE": np.nan, "t": np.nan, "原始均值": np.nan, "原始SD": np.nan}])],
        ignore_index=True)

    # ---------- 表f：清洗口径敏感性 ----------
    sens_rows = []
    variants = {
        "A 原始（含录入错误）": complete_raw,
        "B 整果剔除含>20（主口径）": clean,
        "C 仅置>20格为缺失后逐果保留其余面": None,
        "D 主口径再剔除任一面<5 的果": clean[(clean >= LOW_FLAG).all(axis=1)],
    }
    masked = raw.mask(raw > IMPOSSIBLE_HI)
    variants["C 仅置>20格为缺失后逐果保留其余面"] = masked.dropna()
    for name, df in variants.items():
        vc = variance_components(df.values)
        sens_rows.append({
            "清洗口径": name, "果数": vc["n_fruit"],
            "σ²_果间": vc["var_apple"], "σ²_面间": vc["var_face"],
            "ICC": vc["ICC"], "果内方差占比": vc["within_share"],
            "单面 SD(°Brix)": vc["sd_face"], "5面均值 SEM(°Brix)": vc["sem_mean5"],
            "R²上界@单面真值": vc["var_apple"] / (vc["var_apple"] + vc["var_face"]),
        })
    tab_sens = pd.DataFrame(sens_rows)

    # ---------- 光谱侧配套核验 ----------
    spec = pd.read_csv(os.path.join(DATA_DIR, SPEC_FILE))
    sid = spec["实际苹果编号"].astype(str)
    ent = sid.str.rsplit("-", n=1).str[0]
    wl = [c for c in spec.columns if c.endswith("nm")]
    tab_spec = pd.DataFrame([
        {"项": "光谱记录数", "值": len(spec)},
        {"项": "光谱侧实体数", "值": ent.nunique()},
        {"项": "波段数（README 写 348，实测以此为准）", "值": len(wl)},
        {"项": "波长范围", "值": f"{wl[0]} – {wl[-1]}"},
        {"项": "拥有完整 5 面光谱的果数", "值": int((ent.value_counts() == 5).sum())},
        {"项": "糖度侧与光谱侧实体是否完全一致", "值": str(set(ent) == set(ssc["id"].astype(str)))},
        {"项": "反射率取值范围",
         "值": f"[{np.nanmin(spec[wl].values):.4f}, {np.nanmax(spec[wl].values):.4f}]"},
    ])

    out = write_script_workbook(__file__, {
        0: ("数据质量与异常值判定", tab_quality),
        1: ("样本量流转", flow),
        2: ("方差分解（总体与分产地）", tab_vc),
        3: ("果内异质性幅度 vs 仪器精度", tab_het),
        4: ("可达上界 R²/RMSE", tab_ceiling),
        5: ("采样面位置系统效应", tab_face),
        6: ("清洗口径敏感性", tab_sens),
        7: ("光谱侧结构核验", tab_spec),
    })

    logger.log(f"清洗后果数 n={vc_all['n_fruit']}")
    logger.log(f"σ²_果间={vc_all['var_apple']:.4f}  σ²_面间={vc_all['var_face']:.4f}")
    logger.log(f"ICC={vc_all['ICC']:.4f}  果内方差占比={vc_all['within_share']:.4f} "
               f"CI95 ICC=[{lo_icc:.3f},{hi_icc:.3f}]")
    logger.log(f"单面 SD={vc_all['sd_face']:.4f} °Brix   5面均值 SEM={vc_all['sem_mean5']:.4f} °Brix")
    logger.log(f"果内 SD 中位 {sd_w.median():.4f} = 折光仪精度的 {sd_w.median()/REFRACTOMETER_PRECISION:.1f} 倍")
    r2_1 = vc_all["var_apple"] / (vc_all["var_apple"] + vc_all["var_face"])
    r2_5 = vc_all["var_apple"] / (vc_all["var_apple"] + vc_all["var_face"] / 5)
    logger.log(f"R²上界：单面真值 {r2_1:.4f} / 5面均值真值 {r2_5:.4f}")
    logger.log(f"采样面位置置换检验 p={p_face:.4f}")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
