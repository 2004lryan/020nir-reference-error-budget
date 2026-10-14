"""C4 pilot · 第 2 段（决定性）：局部光谱到底能不能感知局部糖度？

问题
----
93 号脚本给出：若模型只能感知「整果潜在 SSC」μ_i，则以单面破坏性值为参考真值时
R² 上界仅 0.477。但这个上界是否**绑定**，取决于逐面光谱能否感知面级偏离
e_ij = Y_ij − μ_i。本数据集光谱与糖度都是逐面配对的，因此这件事可以**直接测**。

预注册的判定规则（先写后跑，不得事后调整）
------------------------------------------
在按苹果 ID 分组的留出集上，用面级光谱预测 e_ij，记 R²_dev：

  · **分叉 A（天花板绑定）**：R²_dev ≤ 0.10
        → 局部光谱基本感知不到局部糖度差异。单面参考真值下 R² 上界 0.477 是真绑定，
          文献中大量以单点/单面参考值报告的高 R² 在结构上可疑。可构造的修正（多面参考、
          衰减校正、序贯扫描）成为论文主交付物。
  · **分叉 B（天花板不绑定）**：R²_dev > 0.30
        → 局部光谱确实能感知局部糖度。正确结论变成「必须逐面建模、不能用整果标签」，
          这是另一篇论文，须重新立项评估。
  · **中间地带**：0.10 < R²_dev ≤ 0.30 → 部分可感知，两条主张都要弱化，单独定性。

R² 一律用留出集上的  1 − SSE/SST，SST 以**训练集**上该目标的均值为基准（避免用测试集
自身均值虚高）。所有划分按苹果 ID 分组，同一个苹果的 5 个面绝不跨越训练/测试。

对照与消融
----------
预处理：raw / SNV / 一阶导(SG) / SNV+一阶导 —— 化学计量学标准四件套。
目标：① 面级 SSC（文献常规做法）② 整果均值 SSC ③ **面级偏离 e_ij（决定性）**
另做：把 5 个面的光谱平均后预测整果均值（"整果扫描"口径），与单面预测整果均值对比，
      看多面平均对光谱侧的增益，与 93 号脚本算出的标签侧增益（RMSE 下界 1.597→0.714）
      是否同阶。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEED = 2004
N_REPEAT = 20              # 重复随机分组划分次数
TEST_FRAC = 0.40
MAX_PLS = 20               # 成分数上限，训练集内 GroupKFold 选
N_INNER_FOLD = 5
IMPOSSIBLE_HI = 20.0

import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_DIR = os.path.join(DATA_ROOT, "001_apple_hyperspectral_multiyear")
FACE_COLS = ["顶部", "底部", "侧面1", "侧面2", "侧面3"]

# 预注册阈值
THR_A = 0.10
THR_B = 0.30

logger = Logger(__file__)


# --------------------------------------------------------------------------- #
def snv(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(1, keepdims=True)) / (X.std(1, ddof=1, keepdims=True) + 1e-12)


def deriv1(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)


PREPROC = {
    "raw": lambda X: X,
    "SNV": snv,
    "1stDeriv": deriv1,
    "SNV+1stDeriv": lambda X: deriv1(snv(X)),
}


def pls_fit_predict(Xtr, ytr, Xte, groups_tr) -> tuple[np.ndarray, int]:
    """训练集内按苹果分组选成分数，再在全训练集上重拟合并预测测试集。"""
    n_comp_max = min(MAX_PLS, Xtr.shape[1], max(2, len(np.unique(groups_tr)) - 2))
    gkf = GroupKFold(n_splits=min(N_INNER_FOLD, len(np.unique(groups_tr))))
    best_c, best_err = 1, np.inf
    for c in range(1, n_comp_max + 1):
        errs = []
        for itr, iva in gkf.split(Xtr, ytr, groups_tr):
            m = PLSRegression(n_components=c, scale=False)
            m.fit(Xtr[itr], ytr[itr])
            errs.append(np.mean((m.predict(Xtr[iva]).ravel() - ytr[iva]) ** 2))
        e = float(np.mean(errs))
        if e < best_err - 1e-9:
            best_err, best_c = e, c
    m = PLSRegression(n_components=best_c, scale=False)
    m.fit(Xtr, ytr)
    return m.predict(Xte).ravel(), best_c


def r2_holdout(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - train_mean) ** 2)
    return 1.0 - sse / sst if sst > 0 else np.nan


# --------------------------------------------------------------------------- #
def main() -> None:
    logger.log(f"[94localsense] SEED={SEED} N_REPEAT={N_REPEAT} 预注册阈值 A≤{THR_A} B>{THR_B}")

    ssc = pd.read_csv(os.path.join(DATA_DIR, "新疆-山东-甘肃苹果糖度数据2025.csv"), encoding="gbk")
    ssc.columns = ["id"] + FACE_COLS
    raw = ssc[FACE_COLS].apply(pd.to_numeric, errors="coerce")
    keep = raw.drop(index=raw[(raw > IMPOSSIBLE_HI).any(axis=1)].index).dropna()
    ssc_long = (ssc.loc[keep.index, ["id"]].join(keep)
                .melt(id_vars="id", var_name="face", value_name="ssc"))
    ssc_long["key"] = ssc_long["id"].astype(str) + "-" + ssc_long["face"]

    spec = pd.read_csv(os.path.join(DATA_DIR, "新疆-山东-甘肃苹果光谱数据2025.csv"))
    wl_cols = [c for c in spec.columns if c.endswith("nm")]
    spec["key"] = spec["实际苹果编号"].astype(str)

    df = ssc_long.merge(spec[["key"] + wl_cols], on="key", how="inner")
    logger.log(f"配对成功 {len(df)} 个 (面, 光谱, 糖度) 三元组，覆盖 {df['id'].nunique()} 个苹果")

    fruit_mean = df.groupby("id")["ssc"].transform("mean")
    df["y_face"] = df["ssc"]
    df["y_fruit"] = fruit_mean
    df["y_dev"] = df["ssc"] - fruit_mean          # 决定性目标 e_ij

    X_all = df[wl_cols].values.astype(float)
    groups = df["id"].values
    uniq = np.unique(groups)
    wl_nm = np.array([float(c.replace("nm", "")) for c in wl_cols])

    rows = []
    rng = np.random.default_rng(SEED)
    for pname, pfun in PREPROC.items():
        Xp = pfun(X_all)
        for target in ["y_face", "y_fruit", "y_dev"]:
            y = df[target].values.astype(float)
            r2s, rmses, ncs = [], [], []
            for rep in range(N_REPEAT):
                perm = rng.permutation(uniq)
                n_te = int(round(TEST_FRAC * len(uniq)))
                te_ids = set(perm[:n_te])
                m_te = np.array([g in te_ids for g in groups])
                ytr, yte = y[~m_te], y[m_te]
                pred, nc = pls_fit_predict(Xp[~m_te], ytr, Xp[m_te], groups[~m_te])
                r2s.append(r2_holdout(yte, pred, ytr.mean()))
                rmses.append(float(np.sqrt(np.mean((yte - pred) ** 2))))
                ncs.append(nc)
            rows.append({
                "预处理": pname, "目标": target,
                "目标含义": {"y_face": "面级 SSC（文献常规口径）",
                          "y_fruit": "整果均值 SSC",
                          "y_dev": "面级偏离 e_ij ← 决定性"}[target],
                "R2_均值": np.mean(r2s), "R2_SD": np.std(r2s, ddof=1),
                "R2_中位": np.median(r2s),
                "RMSE_均值": np.mean(rmses), "RMSE_SD": np.std(rmses, ddof=1),
                "PLS成分数_中位": float(np.median(ncs)),
            })
            logger.log(f"  {pname:14s} {target:8s} R²={np.mean(r2s):+.4f}±{np.std(r2s,ddof=1):.4f} "
                       f"RMSE={np.mean(rmses):.4f} nc={np.median(ncs):.0f}")
    tab_main = pd.DataFrame(rows)

    # ---------- 单面 vs 五面平均光谱，预测整果均值 ----------
    best_pre = tab_main.loc[tab_main.query("目标=='y_fruit'")["R2_均值"].idxmax(), "预处理"]
    Xb = PREPROC[best_pre](X_all)
    wide = (df.assign(_i=df.groupby("id").cumcount())
              .pivot_table(index="id", columns="face", values="y_face"))
    fruit_ids = wide.index.values
    Xmean = np.vstack([Xb[groups == fid].mean(0) for fid in fruit_ids])
    ymean = np.array([df.loc[df["id"] == fid, "y_fruit"].iloc[0] for fid in fruit_ids])

    agg_rows = []
    rng2 = np.random.default_rng(SEED)
    r2_single, r2_avg = [], []
    for rep in range(N_REPEAT):
        perm = rng2.permutation(fruit_ids)
        n_te = int(round(TEST_FRAC * len(fruit_ids)))
        te = set(perm[:n_te])
        mte_f = np.array([f in te for f in fruit_ids])
        p, _ = pls_fit_predict(Xmean[~mte_f], ymean[~mte_f], Xmean[mte_f], fruit_ids[~mte_f])
        r2_avg.append(r2_holdout(ymean[mte_f], p, ymean[~mte_f].mean()))
        m_te = np.array([g in te for g in groups])
        ps, _ = pls_fit_predict(Xb[~m_te], df["y_fruit"].values[~m_te], Xb[m_te], groups[~m_te])
        r2_single.append(r2_holdout(df["y_fruit"].values[m_te], ps, df["y_fruit"].values[~m_te].mean()))
    agg_rows.append({"口径": f"单面光谱 → 整果均值 SSC（{best_pre}）",
                     "R2_均值": np.mean(r2_single), "R2_SD": np.std(r2_single, ddof=1)})
    agg_rows.append({"口径": f"5 面光谱平均 → 整果均值 SSC（{best_pre}）",
                     "R2_均值": np.mean(r2_avg), "R2_SD": np.std(r2_avg, ddof=1)})
    tab_agg = pd.DataFrame(agg_rows)

    # ---------- 预注册判定 ----------
    dev_best = tab_main.query("目标=='y_dev'")["R2_均值"].max()
    dev_row = tab_main.query("目标=='y_dev'").sort_values("R2_均值", ascending=False).iloc[0]
    if dev_best <= THR_A:
        verdict, meaning = "分叉 A", "局部光谱基本感知不到局部糖度 → 单面参考真值下的 R² 上界 0.477 真绑定"
    elif dev_best > THR_B:
        verdict, meaning = "分叉 B", "局部光谱能感知局部糖度 → 天花板不绑定，主张须改为「必须逐面建模」"
    else:
        verdict, meaning = "中间地带", "部分可感知 → 两条主张都需弱化并单独定性"
    tab_verdict = pd.DataFrame([
        {"项": "预注册阈值", "值": f"A: R²_dev ≤ {THR_A} / B: R²_dev > {THR_B}"},
        {"项": "实测 R²_dev（取各预处理最优）", "值": f"{dev_best:.4f}"},
        {"项": "对应预处理", "值": dev_row["预处理"]},
        {"项": "R²_dev 的 SD", "值": f"{dev_row['R2_SD']:.4f}"},
        {"项": "**判定**", "值": verdict},
        {"项": "含义", "值": meaning},
        {"项": "备注", "值": "R² 以训练集均值为 SST 基准；划分按苹果 ID 分组，"
                            f"重复 {N_REPEAT} 次；PLS 成分数在训练集内 GroupKFold 选"},
    ])

    out = write_script_workbook(__file__, {
        0: ("主表：三个目标 × 四种预处理", tab_main),
        1: ("单面 vs 五面平均光谱", tab_agg),
        2: ("预注册判定", tab_verdict),
    })
    logger.log(f"\n=== 预注册判定：{verdict} ===  R²_dev(best) = {dev_best:.4f}  [{dev_row['预处理']}]")
    logger.log(meaning)
    logger.log(f"单面→整果 R²={np.mean(r2_single):.4f} / 5面平均→整果 R²={np.mean(r2_avg):.4f}")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
