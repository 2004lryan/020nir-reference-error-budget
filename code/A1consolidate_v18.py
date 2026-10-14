"""A1 · 论文数字落盘补全（回应零上下文 claim-audit 的 FAIL）。

审计发现：正文中若干关键数字此前只由临时脚本算出、未写入任何工作簿，导致
「图中每个数值均可追溯」这句话不成立。本脚本把它们全部重算并落盘，使论文中
每一个数字都能指到 `outputs/A1consolidate_v18.xlsx` 的具体表格与单元格。

补齐清单（括号内为正文位置）
  表a  1/m 标度的理论 vs 实测方差与偏差（3.2 第 1 条）
  表b  面间协方差矩阵、非对角均值、复合对称置换检验 p（3.2 第 2 条）
  表c  纯标签基线的理论值与实测值（3.2 第 3 条）—— 理论值按原始方差分量精确计算
  表d  顶部/底部配对 t 检验（3.3）
  表e  仅用 3 个侧面的方差与极差（3.3）
  表f  001 波段子集建模：全谱 / 干净波段 / 860–1001 / 900–1001（4.3）
  表g  012 猕猴桃强模型 R²（DM 与 SSC）（4.3）
  表h  003 DeepHS 按果分组 vs 按采集日分组（3.5）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import ttest_rel
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEED = 2004
N_PERM = 3000
N_REPEAT = 15
TEST_FRAC = 0.40
IMPOSSIBLE_HI = 20.0
import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
D001 = os.path.join(DATA_ROOT, "001_apple_hyperspectral_multiyear")
D012 = os.path.join(DATA_ROOT, "012_kiwifruit_nir_drymatter", "kiwifruit_dat.csv")
TRAJ = "03data/processed/v18_deephs_traj.parquet"
FACE = ["顶部", "底部", "侧面1", "侧面2", "侧面3"]

logger = Logger(__file__)


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, ddof=1, keepdims=True) + 1e-12)


def d1(X):
    return savgol_filter(X, 11, 2, deriv=1, axis=1)


def cv_r2(X, y, groups, nrep=N_REPEAT, seed=SEED, max_comp=24):
    """按 groups 分组的重复留出，PLS 成分数在训练集内再分组 CV 选。"""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    out, rms = [], []
    for _ in range(nrep):
        te = set(rng.permutation(uq)[: int(round(TEST_FRAC * len(uq)))])
        m = np.array([g in te for g in groups])
        Xtr, ytr, gtr = X[~m], y[~m], groups[~m]
        best, berr = 2, np.inf
        inner = list(GroupKFold(n_splits=4).split(Xtr, ytr, gtr))
        for c in range(2, min(max_comp, Xtr.shape[1]) + 1, 2):
            e = [np.mean((PLSRegression(c).fit(Xtr[a], ytr[a]).predict(Xtr[b]).ravel()
                          - ytr[b]) ** 2) for a, b in inner]
            if np.mean(e) < berr - 1e-9:
                berr, best = float(np.mean(e)), c
        p = PLSRegression(best).fit(Xtr, ytr).predict(X[m]).ravel()
        out.append(1 - np.sum((y[m] - p) ** 2) / np.sum((y[m] - ytr.mean()) ** 2))
        rms.append(float(np.sqrt(np.mean((y[m] - p) ** 2))))
    return float(np.mean(out)), float(np.std(out, ddof=1)), float(np.mean(rms))


def main() -> None:
    # ── 001 标签侧 ────────────────────────────────────────────────────────
    ssc = pd.read_csv(os.path.join(D001, "新疆-山东-甘肃苹果糖度数据2025.csv"), encoding="gbk")
    ssc.columns = ["id"] + FACE
    raw = ssc[FACE].apply(pd.to_numeric, errors="coerce")
    clean = raw.drop(index=raw[(raw > IMPOSSIBLE_HI).any(axis=1)].index).dropna()
    V = clean.values
    n, k = V.shape
    mu = V.mean(1)
    MSB = k * ((mu - V.mean()) ** 2).sum() / (n - 1)
    MSW = ((V - mu[:, None]) ** 2).sum() / (n * (k - 1))
    s2a, s2f = max((MSB - MSW) / k, 0.0), MSW
    logger.log(f"n={n} σ²a={s2a:.6f} σ²f={s2f:.6f}")

    # 表a：1/m 标度
    rows = []
    for m in (2, 3, 4, 5):
        obs = float(V[:, :m].mean(1).var(ddof=1))
        th = s2a + s2f / m
        rows.append({"m（取前 m 个面的均值）": m, "理论方差 σ²a+σ²f/m": th,
                     "实测方差": obs, "偏差%": 100 * (obs - th) / th,
                     "说明": "σ²a、σ²f 由 m=5 的完整设计估得，m=2/3/4 属独立外推"
                             if m < 5 else "m=5 为估计所用设计本身，非独立检验"})
    tab_a = pd.DataFrame(rows)

    # 表b：面间协方差与复合对称
    S = np.cov(V, rowvar=False)
    off = S[np.triu_indices(k, 1)]
    rng = np.random.default_rng(SEED)
    obs_disp = float(np.std(off, ddof=1))
    perm = np.empty(N_PERM)
    for b in range(N_PERM):
        Vs = np.apply_along_axis(rng.permutation, 1, V)
        Sp = np.cov(Vs, rowvar=False)
        perm[b] = np.std(Sp[np.triu_indices(k, 1)], ddof=1)
    p_cs = float((perm >= obs_disp).mean())
    tab_b = pd.concat([
        pd.DataFrame(S, index=FACE, columns=FACE).reset_index().rename(columns={"index": "面"}),
        pd.DataFrame([{"面": "—— 汇总 ——"},
                      {"面": "非对角元均值", "顶部": float(off.mean())},
                      {"面": "非对角元 SD", "顶部": obs_disp},
                      {"面": "非对角元最小/最大", "顶部": f"{off.min():.4f} / {off.max():.4f}"},
                      {"面": "可交换性预言值 = σ²a", "顶部": s2a},
                      {"面": f"复合对称置换检验 p（B={N_PERM}）", "顶部": p_cs}])],
        ignore_index=True)

    # 表c：纯标签基线
    loo = (V.sum(1, keepdims=True) - V) / (k - 1)
    r2_loo = float(1 - np.sum((V - loo) ** 2) / np.sum((V - V.mean()) ** 2))
    th_loo = float(1 - (s2f + s2f / (k - 1)) / (s2a + s2f))
    tab_c = pd.DataFrame([
        {"量": "理论值 1−(σ²f+σ²f/4)/(σ²a+σ²f)", "值": th_loo,
         "说明": "无自由参数；由 σ²a、σ²f 直接算出"},
        {"量": "实测值（其余 4 面均值 → 该面）", "值": r2_loo, "说明": ""},
        {"量": "绝对偏差", "值": abs(r2_loo - th_loo), "说明": ""},
    ])

    # 表d：顶 vs 底 配对 t
    tt = ttest_rel(clean["底部"], clean["顶部"])
    d = clean["底部"] - clean["顶部"]
    tab_d = pd.DataFrame([
        {"量": "底部均值", "值": float(clean["底部"].mean())},
        {"量": "顶部均值", "值": float(clean["顶部"].mean())},
        {"量": "差值（底−顶）", "值": float(d.mean())},
        {"量": "配对 t 统计量", "值": float(tt.statistic)},
        {"量": "p 值", "值": float(tt.pvalue)},
        {"量": "n", "值": int(len(d))},
        {"量": "差值 / 果内 SD 中位数", "值": float(d.mean() / clean.std(axis=1, ddof=1).median())},
    ])

    # 表e：仅 3 个侧面
    sides = clean[["侧面1", "侧面2", "侧面3"]].values
    sw = sides - sides.mean(1, keepdims=True)
    var_side = float((sw ** 2).sum() / (len(sides) * 2))
    W = V - V.mean(1, keepdims=True)
    pos = W.mean(0)
    ss_w = float((W ** 2).sum())
    ss_pos = float(n * (pos ** 2).sum())
    tab_e = pd.DataFrame([
        {"量": "仅 3 个侧面估得的面间方差", "值": var_side,
         "说明": "同赤道环带、解剖学等价、不含梗-萼梯度"},
        {"量": "全 5 面估得的面间方差 σ²f", "值": s2f, "说明": ""},
        {"量": "比值", "值": var_side / s2f, "说明": ""},
        {"量": "3 侧面极差中位数", "值": float(np.median(sides.max(1) - sides.min(1))), "说明": ""},
        {"量": "面位置固定效应平方和 SS_pos", "值": ss_pos, "说明": ""},
        {"量": "果内总平方和 SS_within", "值": ss_w, "说明": ""},
        {"量": "位置效应解释果内变异的比例%", "值": 100 * ss_pos / ss_w,
         "说明": "正文 3.3 的 0.95%"},
    ] + [{"量": f"面位置固定效应：{f}", "值": float(v), "说明": ""} for f, v in zip(FACE, pos)])

    # ── 001 光谱侧：波段子集 ──────────────────────────────────────────────
    lg = (ssc.loc[clean.index, ["id"]].join(clean)).melt(id_vars="id", var_name="face",
                                                         value_name="ssc")
    lg["key"] = lg["id"].astype(str) + "-" + lg["face"]
    sp = pd.read_csv(os.path.join(D001, "新疆-山东-甘肃苹果光谱数据2025.csv"))
    wc = [c for c in sp.columns if c.endswith("nm")]
    sp["key"] = sp["实际苹果编号"].astype(str)
    df = lg.merge(sp[["key"] + wc], on="key", how="inner").reset_index(drop=True)
    wl = np.array([float(c[:-2]) for c in wc])
    X = df[wc].values.astype(float)
    g = df["id"].values
    y = df.groupby("id")["ssc"].transform("mean").values.astype(float)
    lo = (X < 0.01).mean(0) * 100
    hi = (X > 0.99).mean(0) * 100
    cleanband = (lo + hi) <= 1
    rows = []
    for tag, mask in (("全谱 346 波段", np.ones(len(wl), bool)),
                      (f"仅无饱和波段（{int(cleanband.sum())} 个）", cleanband),
                      ("860–1001 nm", wl >= 860),
                      ("900–1001 nm（糖倍频带）", wl >= 900)):
        r2, sd, _ = cv_r2(X[:, mask], y, g)
        rows.append({"波段子集": tag, "波段数": int(mask.sum()), "R2_均值": r2, "R2_SD": sd})
        logger.log(f"  001 {tag:26s} R²={r2:+.4f}")
    tab_f = pd.DataFrame(rows)

    # ── 012 猕猴桃强模型 ──────────────────────────────────────────────────
    kd = pd.read_csv(D012)
    meta = ["Dataset", "Date", "device", "sample_id", "SSC", "DM"]
    kw = [c for c in kd.columns if c not in meta]
    kd = kd.dropna(subset=["SSC", "DM"])
    ok = ~np.isnan(kd[kw].values.astype(float)).any(0)
    kw = [c for c, b in zip(kw, ok) if b]
    kwl = np.array([float(c[1:]) for c in kw])
    KX = d1(kd[kw].values.astype(float)[:, kwl >= 700])
    kg = kd.sample_id.values
    rows = []
    for tgt in ("DM", "SSC"):
        r2, sd, rm = cv_r2(KX, kd[tgt].values.astype(float), kg, nrep=6)
        rows.append({"目标": tgt, "配置": "一阶导 700–1065nm，成分数 CV 选，按 sample_id 分组",
                     "R2_均值": r2, "R2_SD": sd, "RMSEP": rm})
        logger.log(f"  012 {tgt} R²={r2:+.4f} RMSEP={rm:.4f}")
    tab_g = pd.DataFrame(rows)
    tab_g_note = pd.DataFrame([
        {"项": "总谱条数（原始）", "值": 11982},
        {"项": "总果数（原始）", "值": 5418},
        {"项": "SSC 或 DM 缺失后剩余谱条数", "值": int(len(kd))},
        {"项": "对应果数", "值": int(kd.sample_id.nunique())},
        {"项": "说明", "值": "正文样本量一律引用原始值 11,982 / 5,418"},
    ])

    # ── 003 DeepHS 按果 vs 按天 ──────────────────────────────────────────
    tj = pd.read_parquet(TRAJ)
    twc = [c for c in tj.columns if c.startswith("w")]
    TX = snv(tj[twc].to_numpy(float))
    ty = tj["day_idx"].to_numpy(float)
    fk = (tj["fruit"] + "|" + tj["sample"]).to_numpy()
    dk = (tj["fruit"] + "|" + tj["day"]).to_numpy()
    rows = []
    for tag, grp, mask in (("全体·按果分组", fk, np.ones(len(tj), bool)),
                           ("全体·按采集日分组", dk, np.ones(len(tj), bool)),
                           ("Kiwi·按果分组", fk, (tj["fruit"] == "Kiwi").values),
                           ("Kiwi·按采集日分组", tj["day"].to_numpy(), (tj["fruit"] == "Kiwi").values),
                           ("Avocado·按果分组", fk, (tj["fruit"] == "Avocado").values),
                           ("Avocado·按采集日分组", tj["day"].to_numpy(),
                            (tj["fruit"] == "Avocado").values)):
        r2, sd, rm = cv_r2(TX[mask], ty[mask], grp[mask], nrep=6, max_comp=20)
        rows.append({"设置": tag, "n谱": int(mask.sum()), "R2_均值": r2, "R2_SD": sd,
                     "RMSE（日序）": rm})
        logger.log(f"  003 {tag:22s} R²={r2:+.4f} RMSE={rm:.2f}")
    tab_h = pd.DataFrame(rows)

    out = write_script_workbook(__file__, {
        0: ("1/m 标度理论vs实测", tab_a),
        1: ("面间协方差与复合对称", tab_b),
        2: ("纯标签基线理论vs实测", tab_c),
        3: ("顶底配对t检验", tab_d),
        4: ("三侧面与位置效应", tab_e),
        5: ("001 波段子集建模", tab_f),
        6: ("012 强模型", tab_g),
        7: ("012 样本量口径", tab_g_note),
        8: ("003 按果vs按天分组", tab_h),
    })
    logger.log(f"\n纯标签基线 理论={th_loo:.6f} 实测={r2_loo:.6f}")
    logger.log(f"非对角均值={off.mean():.6f} vs σ²a={s2a:.6f}；复合对称 p={p_cs:.4f}")
    logger.log(f"顶底配对 t={tt.statistic:.4f}")
    logger.log(f"3 侧面方差={var_side:.6f}（占全 5 面 {100*var_side/s2f:.2f}%）")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
