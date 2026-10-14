"""C4 pilot · 第 3 段：把「模型不行」与「验证协议泄漏」分开。

为什么必须做这一步
------------------
94 号脚本给出：按苹果 ID 分组留出时，单面光谱预测整果均值 SSC 的 R² 只有约 0.12，
远低于文献常年报告的 0.7–0.9。这时有两种互斥解释，不区分开就什么都不能主张：

  解释 (A) 本数据集的光谱质量差 / 我方模型没调好  → 那么 93/94 的"天花板"结论
           会被一句"是你模型不行"废掉，不能发。
  解释 (B) 文献的高 R² 来自验证协议——按行随机划分时，同一个苹果的其余 4 个面
           落进训练集，果水平信息泄漏，测试集不再独立。

区分方法：**同一份数据、同一个模型、同一套超参搜索，只改划分方式。**
若随机划分显著高于分组划分，差值就是泄漏量，(B) 成立；若两者都低，(A) 成立。

附带的诊断（都是为了先把 (A) 排干净，再谈 (B)）
--------------------------------------------
  · 光谱是否饱和/截断（反射率贴 0 或贴 1 的比例）
  · 波段子区间：全谱 vs 700–1001 nm（糖的倍频吸收主要在长波端）
  · 分产地单独建模（合并三产地可能引入批次异质，压低性能）
  · 用**果均值光谱**预测果均值 SSC（去掉面级噪声后的最优情形）
  · 与"用同果其余 4 面的 SSC 均值预测该面 SSC"这个非光谱基线对照——
    后者是纯标签侧信息，给出"果水平信息本身能到多少"的参照点

一律用与 94 号相同的 PLS 设定与 R² 定义（SST 以训练集均值为基准）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEED = 2004
N_REPEAT = 20
TEST_FRAC = 0.40
MAX_PLS = 20
N_INNER_FOLD = 5
IMPOSSIBLE_HI = 20.0

import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_DIR = os.path.join(DATA_ROOT, "001_apple_hyperspectral_multiyear")
FACE_COLS = ["顶部", "底部", "侧面1", "侧面2", "侧面3"]

logger = Logger(__file__)


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, ddof=1, keepdims=True) + 1e-12)


def deriv1(X):
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)


def fit_predict(Xtr, ytr, Xte, groups_tr=None):
    n_comp_max = min(MAX_PLS, Xtr.shape[1], max(2, Xtr.shape[0] // 10))
    if groups_tr is not None:
        splitter = GroupKFold(n_splits=min(N_INNER_FOLD, len(np.unique(groups_tr))))
        folds = list(splitter.split(Xtr, ytr, groups_tr))
    else:
        folds = list(KFold(n_splits=N_INNER_FOLD, shuffle=True,
                           random_state=SEED).split(Xtr))
    best_c, best_err = 1, np.inf
    for c in range(1, n_comp_max + 1):
        errs = []
        for itr, iva in folds:
            m = PLSRegression(n_components=c, scale=False)
            m.fit(Xtr[itr], ytr[itr])
            errs.append(np.mean((m.predict(Xtr[iva]).ravel() - ytr[iva]) ** 2))
        e = float(np.mean(errs))
        if e < best_err - 1e-9:
            best_err, best_c = e, c
    m = PLSRegression(n_components=best_c, scale=False)
    m.fit(Xtr, ytr)
    return m.predict(Xte).ravel(), best_c


def r2h(yt, yp, tr_mean):
    return 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - tr_mean) ** 2)


def main():
    logger.log(f"[95splitleak] SEED={SEED} N_REPEAT={N_REPEAT}")

    ssc = pd.read_csv(os.path.join(DATA_DIR, "新疆-山东-甘肃苹果糖度数据2025.csv"), encoding="gbk")
    ssc.columns = ["id"] + FACE_COLS
    raw = ssc[FACE_COLS].apply(pd.to_numeric, errors="coerce")
    keep = raw.drop(index=raw[(raw > IMPOSSIBLE_HI).any(axis=1)].index).dropna()
    long = (ssc.loc[keep.index, ["id"]].join(keep)
            .melt(id_vars="id", var_name="face", value_name="ssc"))
    long["key"] = long["id"].astype(str) + "-" + long["face"]

    spec = pd.read_csv(os.path.join(DATA_DIR, "新疆-山东-甘肃苹果光谱数据2025.csv"))
    wl_cols = [c for c in spec.columns if c.endswith("nm")]
    spec["key"] = spec["实际苹果编号"].astype(str)
    df = long.merge(spec[["key"] + wl_cols], on="key", how="inner").reset_index(drop=True)
    df["origin"] = df["id"].astype(str).str[:2]
    df["y_fruit"] = df.groupby("id")["ssc"].transform("mean")

    X = df[wl_cols].values.astype(float)
    wl = np.array([float(c.replace("nm", "")) for c in wl_cols])
    groups = df["id"].values
    logger.log(f"配对 {len(df)} 行 / {df['id'].nunique()} 果 / {len(wl_cols)} 波段")

    # ---------- 表a：光谱质量诊断（先把"数据坏"这个解释排干净） ----------
    diag = pd.DataFrame([
        {"诊断项": "反射率 = 0 的格子占比(%)", "值": 100 * np.mean(X <= 0)},
        {"诊断项": "反射率 = 1 的格子占比(%)", "值": 100 * np.mean(X >= 1)},
        {"诊断项": "反射率 <0.01 的格子占比(%)", "值": 100 * np.mean(X < 0.01)},
        {"诊断项": "反射率 >0.99 的格子占比(%)", "值": 100 * np.mean(X > 0.99)},
        {"诊断项": "含任一饱和点的光谱条数占比(%)",
         "值": 100 * np.mean((X <= 0).any(1) | (X >= 1).any(1))},
        {"诊断项": "光谱曲线两两相关中位数（同产地内）",
         "值": float(np.median(np.corrcoef(X[:200])[np.triu_indices(200, 1)]))},
        {"诊断项": "单波段与 SSC 的最大 |Pearson r|",
         "值": float(np.max(np.abs([np.corrcoef(X[:, j], df["ssc"])[0, 1]
                                    for j in range(X.shape[1])])))},
        {"诊断项": "最大 |r| 对应波长(nm)",
         "值": float(wl[int(np.argmax(np.abs([np.corrcoef(X[:, j], df["ssc"])[0, 1]
                                              for j in range(X.shape[1])])))])},
        {"诊断项": "一阶导后单波段与 SSC 的最大 |r|",
         "值": float(np.max(np.abs([np.corrcoef(deriv1(X)[:, j], df["ssc"])[0, 1]
                                    for j in range(X.shape[1])])))},
    ])

    # ---------- 表b：核心对照 —— 分组划分 vs 随机划分 ----------
    rows = []
    Xr, Xd = X, deriv1(X)
    for pre_name, Xp in (("raw", Xr), ("1stDeriv", Xd)):
        for target in ("ssc", "y_fruit"):
            y = df[target].values.astype(float)
            for split in ("按苹果分组（正确）", "按行随机（文献常见）"):
                r2s, rmses = [], []
                rng = np.random.default_rng(SEED)
                for _ in range(N_REPEAT):
                    if split.startswith("按苹果"):
                        uq = np.unique(groups)
                        te_ids = set(rng.permutation(uq)[:int(round(TEST_FRAC * len(uq)))])
                        mte = np.array([g in te_ids for g in groups])
                        gtr = groups[~mte]
                    else:
                        idx = rng.permutation(len(df))
                        mte = np.zeros(len(df), bool)
                        mte[idx[:int(round(TEST_FRAC * len(df)))]] = True
                        gtr = None
                    p, _ = fit_predict(Xp[~mte], y[~mte], Xp[mte], gtr)
                    r2s.append(r2h(y[mte], p, y[~mte].mean()))
                    rmses.append(float(np.sqrt(np.mean((y[mte] - p) ** 2))))
                rows.append({"预处理": pre_name,
                             "目标": "面级 SSC" if target == "ssc" else "整果均值 SSC",
                             "划分方式": split,
                             "R2_均值": np.mean(r2s), "R2_SD": np.std(r2s, ddof=1),
                             "RMSE_均值": np.mean(rmses)})
                logger.log(f"  {pre_name:9s} {target:8s} {split:16s} "
                           f"R²={np.mean(r2s):+.4f}±{np.std(r2s,ddof=1):.4f}")
    tab_split = pd.DataFrame(rows)

    # 泄漏量
    leak = []
    for pre_name in ("raw", "1stDeriv"):
        for tgt in ("面级 SSC", "整果均值 SSC"):
            g = tab_split.query("预处理==@pre_name and 目标==@tgt and 划分方式.str.startswith('按苹果')")["R2_均值"].iloc[0]
            r = tab_split.query("预处理==@pre_name and 目标==@tgt and 划分方式.str.startswith('按行')")["R2_均值"].iloc[0]
            leak.append({"预处理": pre_name, "目标": tgt, "分组划分 R²": g,
                         "随机划分 R²": r, "泄漏量(绝对)": r - g,
                         "虚高倍数": (r / g) if g > 0 else np.nan})
    tab_leak = pd.DataFrame(leak)

    # ---------- 表c：分产地 / 波段子区间 / 果均值光谱 ----------
    extra = []
    for name, mask, Xuse in [
        ("全体·全谱", np.ones(len(df), bool), Xd),
        ("全体·700–1001nm", np.ones(len(df), bool), Xd[:, wl >= 700]),
        *[(f"{o}·全谱", (df['origin'] == o).values, Xd) for o in sorted(df['origin'].unique())],
    ]:
        y = df["y_fruit"].values[mask]
        gsub = groups[mask]
        Xs = Xuse[mask]
        r2s = []
        rng = np.random.default_rng(SEED)
        for _ in range(N_REPEAT):
            uq = np.unique(gsub)
            te = set(rng.permutation(uq)[:int(round(TEST_FRAC * len(uq)))])
            mte = np.array([g in te for g in gsub])
            p, _ = fit_predict(Xs[~mte], y[~mte], Xs[mte], gsub[~mte])
            r2s.append(r2h(y[mte], p, y[~mte].mean()))
        extra.append({"设置": name, "n行": int(mask.sum()), "n果": len(np.unique(gsub)),
                      "R2_均值": np.mean(r2s), "R2_SD": np.std(r2s, ddof=1)})
        logger.log(f"  [子集] {name:18s} R²={np.mean(r2s):+.4f}")
    tab_extra = pd.DataFrame(extra)

    # ---------- 表d：非光谱基线 —— 同果其余 4 面能预测该面多少 ----------
    v = keep.values
    n, k = v.shape
    loo_pred = (v.sum(1, keepdims=True) - v) / (k - 1)
    r2_loo = 1 - np.sum((v - loo_pred) ** 2) / np.sum((v - v.mean()) ** 2)
    tab_base = pd.DataFrame([
        {"基线": "同果其余 4 面均值 → 该面 SSC（纯标签侧，无光谱）", "R2": r2_loo,
         "说明": "果水平信息的天然参照点；理论值≈ICC，实测 ICC=0.477"},
        {"基线": "全局均值 → 该面 SSC", "R2": 0.0, "说明": "定义上为 0"},
    ])

    out = write_script_workbook(__file__, {
        0: ("光谱质量诊断", diag),
        1: ("核心对照：分组 vs 随机划分", tab_split),
        2: ("泄漏量", tab_leak),
        3: ("子集与波段区间", tab_extra),
        4: ("非光谱基线", tab_base),
    })
    logger.log("\n=== 泄漏量 ===")
    for _, r in tab_leak.iterrows():
        logger.log(f"  {r['预处理']:9s} {r['目标']:10s} 分组 {r['分组划分 R²']:+.4f} → "
                   f"随机 {r['随机划分 R²']:+.4f}  Δ={r['泄漏量(绝对)']:+.4f} "
                   f"({r['虚高倍数']:.2f}×)")
    logger.log(f"非光谱基线 R²（其余4面→该面）= {r2_loo:.4f}  vs  ICC=0.477")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
