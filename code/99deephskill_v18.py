"""C3 pilot · 第 2 段：消费 98 号的轨迹数据集，判定预注册 kill ② 与 kill ③。

预注册 kill criterion（与 98 号同一套，先写后跑，不得事后调整）
------------------------------------------------------------
  kill ②「无共同单调路径」：
        扣除每颗果自身均值后做 PCA，PC1 得分与成像日序的 Spearman |ρ|，
        **各果中位数 < 0.5** 即触发。
  kill ③「单次扫描不能测龄」：
        按果分组留出，用单条谱预测该谱所在日序，**R² < 0.3** 即触发。

为什么必须扣果均值再做 PCA
------------------------
不扣的话，PC1 几乎必然是"果与果之间的差异"（品种/初始成熟度/大小），
而 C3 要问的是"**同一颗果随时间往哪个方向走**"。扣掉果均值后剩下的就是果内时间变化，
PC1 才是候选的"共同成熟方向"。

比 PC1 更直接的检验：共同方向性
----------------------------
对每颗果算首末两日的位移向量，两两求余弦相似度。若真存在共同路径，
不同果的位移方向应显著同向（平均余弦 ≫ 0）；若各走各的，平均余弦 ≈ 0。
这一条不依赖任何降维假设，作为 kill ② 的佐证同时报告。

预处理
------
raw 与 SNV 两套都跑。SNV 逐谱归一化，能吃掉摆位/光照造成的乘性散射差异——
98 号已证日间变化中采集噪声占比不小，故 SNV 后的结果才是对 C3 更有利的一版；
两版都报，不挑好的报。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEED = 2004
TRAJ = "03data/processed/v18_deephs_traj.parquet"
KILL2_RHO = 0.5
KILL3_R2 = 0.3
MIN_DAYS = 3
MAX_PLS = 20

logger = Logger(__file__)


def snv(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(1, keepdims=True)) / (X.std(1, ddof=1, keepdims=True) + 1e-12)


def kill2(df: pd.DataFrame, W: np.ndarray, tag: str) -> tuple[pd.DataFrame, float, float]:
    """扣果均值 → PCA → PC1 得分 vs 日序 Spearman；外加位移方向余弦。"""
    key = df["fruit"] + "|" + df["sample"]
    ok = key.map(df.groupby(key)["day_idx"].nunique()) >= MIN_DAYS
    W, key, day = W[ok.values], key[ok.values], df.loc[ok.values, "day_idx"].to_numpy()
    # 扣每颗果的均值 → 只剩果内时间变化
    fm = pd.DataFrame(W, index=key.values).groupby(level=0).transform("mean").to_numpy()
    Wc = W - fm
    Wc -= Wc.mean(0)
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    pc1 = Wc @ Vt[0]
    evr = float(S[0] ** 2 / (S ** 2).sum())

    rhos = []
    for k in key.unique():
        m = (key == k).values
        if len(np.unique(day[m])) < MIN_DAYS:
            continue
        r = spearmanr(pc1[m], day[m]).statistic
        if np.isfinite(r):
            rhos.append(abs(float(r)))
    med_rho = float(np.median(rhos)) if rhos else np.nan

    # 位移方向共同性：每果 (末日均值 − 首日均值)，两两余弦
    disp = []
    for k in key.unique():
        m = (key == k).values
        d, w = day[m], Wc[m]
        if len(np.unique(d)) < 2:
            continue
        v = w[d == d.max()].mean(0) - w[d == d.min()].mean(0)
        n = np.linalg.norm(v)
        if n > 0:
            disp.append(v / n)
    D = np.vstack(disp) if disp else np.zeros((0, W.shape[1]))
    if len(D) >= 2:
        C = D @ D.T
        cos_mean = float(C[np.triu_indices(len(D), 1)].mean())
    else:
        cos_mean = np.nan

    tab = pd.DataFrame([
        {"预处理": tag, "项": "参与实体数（≥%d 天）" % MIN_DAYS, "值": int(key.nunique())},
        {"预处理": tag, "项": "扣果均值后 PC1 解释方差比", "值": evr},
        {"预处理": tag, "项": "PC1 得分 vs 日序 |Spearman ρ| 中位", "值": med_rho},
        {"预处理": tag, "项": "  同上 P25 / P75",
         "值": f"{np.percentile(rhos,25):.3f} / {np.percentile(rhos,75):.3f}" if rhos else "—"},
        {"预处理": tag, "项": "位移方向两两余弦均值（共同路径直接检验）", "值": cos_mean},
        {"预处理": tag, "项": f"预注册阈值 kill ②（|ρ| 中位 < {KILL2_RHO} 触发）", "值": KILL2_RHO},
        {"预处理": tag, "项": "**kill② 判定**",
         "值": "触发（无共同单调路径）" if med_rho < KILL2_RHO else "通过"},
    ])
    return tab, med_rho, cos_mean


def kill3(df: pd.DataFrame, W: np.ndarray, tag: str) -> tuple[pd.DataFrame, float]:
    """按果分组留出，单条谱 → 日序。"""
    key = (df["fruit"] + "|" + df["sample"]).to_numpy()
    y = df["day_idx"].to_numpy(dtype=float)
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(W, y, groups=key):
        best, berr = 2, np.inf
        inner = list(GroupKFold(n_splits=4).split(W[tr], y[tr], key[tr]))
        for c in range(2, MAX_PLS + 1, 2):
            e = [np.mean((PLSRegression(c).fit(W[tr][a], y[tr][a])
                          .predict(W[tr][b]).ravel() - y[tr][b]) ** 2) for a, b in inner]
            if np.mean(e) < berr - 1e-9:
                berr, best = float(np.mean(e)), c
        pred[te] = PLSRegression(best).fit(W[tr], y[tr]).predict(W[te]).ravel()
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    tab = pd.DataFrame([
        {"预处理": tag, "项": "按果分组留出 R²（单谱 → 日序）", "值": r2},
        {"预处理": tag, "项": "RMSE（日序单位）", "值": rmse},
        {"预处理": tag, "项": "日序标准差（基线）", "值": float(y.std(ddof=1))},
        {"预处理": tag, "项": f"预注册阈值 kill ③（R² < {KILL3_R2} 触发）", "值": KILL3_R2},
        {"预处理": tag, "项": "**kill③ 判定**",
         "值": "触发（单次扫描不能测龄）" if r2 < KILL3_R2 else "通过"},
    ])
    return tab, r2


def main() -> None:
    df = pd.read_parquet(TRAJ)
    wcols = [c for c in df.columns if c.startswith("w")]
    logger.log(f"[99deephskill] 轨迹集 {df.shape}，波段 {len(wcols)}")
    logger.log(f"物种: {dict(df['fruit'].value_counts())}")

    X = df[wcols].to_numpy(dtype=float)
    tabs2, tabs3, summary = [], [], []
    for tag, W in (("raw", X), ("SNV", snv(X))):
        t2, rho, cos_m = kill2(df, W, tag)
        t3, r2 = kill3(df, W, tag)
        tabs2.append(t2)
        tabs3.append(t3)
        summary.append({"预处理": tag, "kill② |ρ|中位": rho, "位移余弦均值": cos_m,
                        "kill③ R²": r2,
                        "kill②": "触发" if rho < KILL2_RHO else "通过",
                        "kill③": "触发" if r2 < KILL3_R2 else "通过"})
        logger.log(f"  [{tag}] kill② |ρ|中位={rho:.3f} 位移余弦={cos_m:.3f} | "
                   f"kill③ R²={r2:.3f}")

    # 分物种（不同物种后熟机制不同，合并可能互相抵消）
    rows_sp = []
    for sp, g in df.groupby("fruit"):
        if g["sample"].nunique() < 10:
            continue
        Ws = snv(g[wcols].to_numpy(dtype=float))
        t2, rho, cos_m = kill2(g.reset_index(drop=True), Ws, sp)
        try:
            _, r2 = kill3(g.reset_index(drop=True), Ws, sp)
        except Exception as exc:                                   # 组数不足等
            r2 = np.nan
            logger.log(f"  [{sp}] kill③ 跳过: {type(exc).__name__}")
        rows_sp.append({"物种": sp, "实体数": int(g["sample"].nunique()),
                        "谱条数": len(g), "kill② |ρ|中位": rho,
                        "位移余弦均值": cos_m, "kill③ R²": r2})
        logger.log(f"  [{sp}] |ρ|中位={rho:.3f} 余弦={cos_m:.3f} R²={r2:.3f}")

    tab_sum = pd.DataFrame(summary)
    out = write_script_workbook(__file__, {
        0: ("预注册判定汇总", tab_sum),
        1: ("kill② 共同路径", pd.concat(tabs2, ignore_index=True)),
        2: ("kill③ 单扫描测龄", pd.concat(tabs3, ignore_index=True)),
        3: ("分物种", pd.DataFrame(rows_sp)),
    })
    fired = [k for k in ("kill②", "kill③")
             if any(r[k] == "触发" for r in summary)]
    logger.log("\n=== 预注册总判定 ===")
    logger.log("触发: " + (", ".join(fired) if fired else "无"))
    logger.log("→ C3 " + ("不立项（如实记录，不放宽阈值）" if fired else "通过前两道 kill，可进入设计"))
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
