#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A2forensicfix_v18 — 修复独立一致性审计查出的两条实证发现。

该审计以零上下文、只读方式对代码与结果做交叉核对，逐条比对稿件断言与工作簿
单元格；两条发现的原始记录见 docs/claim_evidence_map.md。

────────────────────────────────────────────────────────────────────────
【修复 1】EF-001（critical, HP-PHANTOM-RESULT）
    衰减比是论文六个点预测之一、也是三项交付物之一，却**从未被任何分析脚本
    计算或落盘**：数值只硬编码在 B0figures_v18.py:59 的常量 ("衰减比", 0.477, 0.514)，
    而 96satdiag_v18.py:40 的 docstring 写的是 0.515（同句还留着已作废的纯标签
    基线 0.3464）。本脚本按 94localsense 的完全相同口径重算并导出，另给出
    重复级配对自助置信区间——此前从未有过 CI。

【修复 2】EF-003（major, HP-SCOPE-INFLATE）
    A1consolidate 表i 以 day_idx 为回归目标，而分组也用 day。按天留出时，被留出
    那天的目标值在训练集中整档缺失，故 RMSE 1.53→5.47 混合了两件事：
      (a) 去除同日采集的共享干扰（我们主张的泄漏效应）
      (b) 外推到训练范围外的目标值（与泄漏无关的机械退化）
    本脚本改用 DeepHS 的**真实破坏性测量标签** firmness / storage_days 复跑。
    已核：这两个标签在同一 day 内均取多个值（不是 day 的函数），故按天分组
    不再等于抽掉整档目标，混淆被移除。

────────────────────────────────────────────────────────────────────────
【预注册判据 — 写死于运行之前，不得事后调整】

  V7（衰减比可追溯）：导出的实测衰减比与 B0figures 硬编码的 0.514 相差 < 0.001
      → 可追溯性修复成立；否则以本表为准订正正文与图。

  V8（按天分组效应是否真实）：在**真实标签**上
      RMSE(按采集日分组) / RMSE(按果分组) ≥ 1.5
        → 日级泄漏效应在去掉目标外推混淆后依然成立，正文主张保留，
          并把本对照作为其支撑证据一并报告；
      < 1.5
        → 原 3.6 倍主要由目标外推造成，摘要"两个数据集的两种不同泄漏机制"
          **必须撤回**，降级表述为"按天分组敏感性"。

  判据按物种分别评估（Kiwi / Avocado 天数与样本量足够）；汇总口径一并报告
  但不单独作判据，避免物种混合掩盖异质性。

输出：outputs/A2forensicfix_v18.xlsx
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

# ── 与 94localsense_v18.py 完全一致的口径（勿改）────────────────────────
SEED = 2004
N_REPEAT = 20
TEST_FRAC = 0.40
MAX_PLS = 20
N_INNER_FOLD = 4
IMPOSSIBLE_HI = 20.0

# ── 预注册阈值 ─────────────────────────────────────────────────────────
V7_TOL = 0.001
V8_RATIO = 1.5
N_BOOT = 2000

import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_DIR = os.path.join(DATA_ROOT, "001_apple_hyperspectral_multiyear")
FACE_COLS = ["顶部", "底部", "侧面1", "侧面2", "侧面3"]
TRAJ = "03data/processed/v18_deephs_traj.parquet"

logger = Logger(__file__)


def snv(X: np.ndarray) -> np.ndarray:
    m = X.mean(1, keepdims=True)
    s = X.std(1, ddof=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


def pls_fit_predict(Xtr, ytr, Xte, groups_tr):
    """与 94localsense 逐行同构：训练集内按组选成分数，再全训练集重拟合。"""
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


def r2_holdout(y_true, y_pred, train_mean):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - train_mean) ** 2)


def load_apple() -> tuple[pd.DataFrame, list[str]]:
    """与 94localsense main() 前半段逐行同构的载入。"""
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
    fruit_mean = df.groupby("id")["ssc"].transform("mean")
    df["y_face"] = df["ssc"]
    df["y_fruit"] = fruit_mean
    return df, wl_cols


# ══════════════════════════════════════════════════════════════════════
# 修复 1 — 衰减比：真正计算并落盘（含重复级 CI）
# ══════════════════════════════════════════════════════════════════════
def fix1_attenuation_ratio():
    df, wl_cols = load_apple()
    logger.log(f"[修复1] 配对 {len(df)} 个三元组，覆盖 {df['id'].nunique()} 个苹果")
    X = df[wl_cols].values.astype(float)          # raw：与 94 主表口径一致
    groups = df["id"].values
    uniq = np.unique(groups)
    n_te = int(round(TEST_FRAC * len(uniq)))

    rng = np.random.default_rng(SEED)
    r2_face_reps, r2_fruit_reps = [], []
    for _ in range(N_REPEAT):
        te = set(rng.permutation(uniq)[:n_te])
        m = np.array([g in te for g in groups])
        for target, bucket in (("y_face", r2_face_reps), ("y_fruit", r2_fruit_reps)):
            y = df[target].values.astype(float)
            p, _ = pls_fit_predict(X[~m], y[~m], X[m], groups[~m])
            bucket.append(r2_holdout(y[m], p, y[~m].mean()))

    r2_face = float(np.mean(r2_face_reps))
    r2_fruit = float(np.mean(r2_fruit_reps))
    ratio = r2_face / r2_fruit

    reps = np.array([r2_face_reps, r2_fruit_reps])
    bs = np.random.default_rng(SEED)
    boots = [reps[0, i].mean() / reps[1, i].mean()
             for i in (bs.integers(0, reps.shape[1], reps.shape[1]) for _ in range(N_BOOT))]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    logger.log(f"[修复1] R²(面级)={r2_face:.6f}  R²(整果均值)={r2_fruit:.6f}")
    logger.log(f"[修复1] 衰减比实测={ratio:.6f}  CI95=[{lo:.4f}, {hi:.4f}]  理论=ICC=0.476995")

    # ── 两个候选理论值 ────────────────────────────────────────────────
    # 只感知整果水平信息的模型，其可解释方差（分子）对两个目标相同，
    # 故 R²_face / R²_fruit = Var(y_fruit) / Var(y_face)。
    #   y_face  的方差 = σ²a + σ²f          （= 93 表e m=1 的 var_of_truth）
    #   y_fruit 的方差 = σ²a + σ²f/5        （= 93 表e m=5 的 var_of_truth；5 面均值仍带噪）
    VAR_FACE = 2.327025 + 2.551483          # 4.878508
    VAR_FRUIT5 = 2.327025 + 2.551483 / 5    # 2.837321
    THEORY_CORRECT = VAR_FRUIT5 / VAR_FACE  # 0.581596 —— 本文所用 5 面均值目标下的正确预言
    THEORY_ICC = 2.327025 / VAR_FACE        # 0.476995 —— 仅当整果目标无噪(m→∞)才成立

    hardcoded = 0.514
    v7 = abs(ratio - hardcoded) < V7_TOL
    in_ci = lambda v: lo <= v <= hi
    logger.log(f"[修复1] V7（与硬编码 {hardcoded} 比）: |{ratio:.6f}-{hardcoded}| < {V7_TOL} ? -> "
               f"{'通过' if v7 else '未通过 → 以本表为准订正正文与图'}")
    logger.log(f"[修复1] 理论值甄别：ICC={THEORY_ICC:.6f} 在CI内? {in_ci(THEORY_ICC)}  |  "
               f"正确预言(5面均值目标)={THEORY_CORRECT:.6f} 在CI内? {in_ci(THEORY_CORRECT)}")

    tab = pd.DataFrame([
        {"量": "R² · 目标=面级 SSC（y_face, raw）", "值": r2_face,
         "说明": f"{N_REPEAT} 次按果分组留出均值；**两个目标共用同一划分**（配对设计）"},
        {"量": "R² · 目标=5 面均值 SSC（y_fruit, raw）", "值": r2_fruit, "说明": "同上，同一划分"},
        {"量": "衰减比 实测 = R²_face / R²_fruit", "值": ratio,
         "说明": "★ 论文第六个点预测的实测值，此前从未落盘（EF-001）"},
        {"量": "衰减比 CI95 下限", "值": float(lo), "说明": f"重复级配对自助 B={N_BOOT}"},
        {"量": "衰减比 CI95 上限", "值": float(hi), "说明": "本文首次为该比值给出区间"},
        {"量": "理论值A：ICC = σ²a/(σ²a+σ²f)", "值": THEORY_ICC,
         "说明": f"稿中原用值。仅当整果目标无噪(m→∞)成立。在CI内？{in_ci(THEORY_ICC)}"},
        {"量": "理论值B：(σ²a+σ²f/5)/(σ²a+σ²f)", "值": THEORY_CORRECT,
         "说明": f"★ 本文实际用 5 面均值作目标，正确预言应为此值。在CI内？{in_ci(THEORY_CORRECT)}"},
        {"量": "B0figures 原硬编码实测值", "值": hardcoded,
         "说明": "EF-001 指其无结果文件支撑；且其两个目标取自 RNG 序列中不同划分，非配对"},
        {"量": "V7 判定（1=通过）", "值": float(v7), "说明": f"容差 {V7_TOL}"},
    ])
    return tab, ratio, v7


# ══════════════════════════════════════════════════════════════════════
# 修复 2 — 按天分组：改用真实标签，移除目标外推混淆
# ══════════════════════════════════════════════════════════════════════
def cv_grouped(X, y, groups, nrep=N_REPEAT, seed=SEED):
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    n_te = max(1, int(round(TEST_FRAC * len(uq))))
    r2s, rmses = [], []
    for _ in range(nrep):
        te = set(rng.permutation(uq)[:n_te])
        m = np.array([g in te for g in groups])
        if m.sum() < 3 or (~m).sum() < 10 or len(np.unique(groups[~m])) < N_INNER_FOLD:
            continue
        p, _ = pls_fit_predict(X[~m], y[~m], X[m], groups[~m])
        r2s.append(r2_holdout(y[m], p, y[~m].mean()))
        rmses.append(float(np.sqrt(np.mean((y[m] - p) ** 2))))
    if not r2s:
        return np.nan, np.nan, np.nan, 0
    return float(np.mean(r2s)), float(np.std(r2s, ddof=1)), float(np.mean(rmses)), len(r2s)


def fix2_day_control():
    tj = pd.read_parquet(TRAJ)
    wcols = [c for c in tj.columns if c.startswith("w")]
    rows = []
    for target in ("firmness", "storage_days"):
        sub_all = tj.dropna(subset=[target])
        for fruit in ("Kiwi", "Avocado", "__汇总__"):
            sub = sub_all if fruit == "__汇总__" else sub_all[sub_all.fruit == fruit]
            if len(sub) < 40:
                logger.log(f"[修复2] 跳过 {target}/{fruit}：n={len(sub)} 不足")
                continue
            X = snv(sub[wcols].to_numpy(float))
            y = sub[target].to_numpy(float)
            fk = (sub["fruit"] + "|" + sub["sample"]).to_numpy()
            dk = (sub["fruit"] + "|" + sub["day"]).to_numpy()
            confounded = bool((sub.groupby("day")[target].nunique() <= 1).all())
            for tag, grp in (("按果分组", fk), ("按采集日分组", dk)):
                r2, sd, rm, nok = cv_grouped(X, y, grp)
                rows.append({"目标": target, "物种": fruit, "分组": tag,
                             "n谱": len(sub), "n组": int(len(np.unique(grp))),
                             "R2_均值": r2, "R2_SD": sd, "RMSE": rm, "有效重复数": nok,
                             "目标是分组的函数？": "是（混淆）" if confounded else "否（无混淆）"})
            logger.log(f"[修复2] {target:13s} {fruit:9s} n={len(sub):4d} 完成")
    tab = pd.DataFrame(rows)

    verdicts = []
    for (target, fruit), g in tab.groupby(["目标", "物种"]):
        gf = g[g["分组"] == "按果分组"]
        gd = g[g["分组"] == "按采集日分组"]
        if gf.empty or gd.empty:
            continue
        rf, rd = float(gf["RMSE"].iloc[0]), float(gd["RMSE"].iloc[0])
        ratio = rd / rf if np.isfinite(rf) and rf > 0 else np.nan
        ok = bool(np.isfinite(ratio) and ratio >= V8_RATIO)
        verdicts.append({"目标": target, "物种": fruit, "RMSE_按果": rf, "RMSE_按天": rd,
                         "比值": ratio, "V8 阈值": V8_RATIO,
                         "判定": "泄漏主张成立（去混淆后仍达标）" if ok else
                                 "★ 未达标 → 撤回「第二种泄漏机制」，降级为按天分组敏感性"})
        logger.log(f"[修复2] V8 {target}/{fruit}: {rd:.3f}/{rf:.3f} = {ratio:.3f} "
                   f"{'≥' if ok else '<'} {V8_RATIO}")
    return tab, pd.DataFrame(verdicts)


def main() -> None:
    logger.log("=" * 70)
    logger.log("A2forensicfix_v18 — 修复 独立一致性审计 EF-001 / EF-003")
    logger.log(f"预注册：V7_TOL={V7_TOL} V8_RATIO={V8_RATIO} SEED={SEED} N_REPEAT={N_REPEAT}")
    logger.log("=" * 70)

    tab1, ratio, v7 = fix1_attenuation_ratio()
    tab2, tab3 = fix2_day_control()

    out = write_script_workbook(__file__, {
        0: ("衰减比（EF-001 修复）", tab1),
        1: ("真实标签下按果vs按天（EF-003 对照）", tab2),
        2: ("V8 预注册判定", tab3),
    })
    logger.log("=" * 70)
    logger.log("V7（衰减比可追溯）：" + ("通过" if v7 else "未通过 → 以本表为准订正正文与图"))
    if len(tab3):
        fails = tab3[tab3["判定"].str.startswith("★")]
        logger.log(f"V8：{len(tab3) - len(fails)}/{len(tab3)} 个组合达标")
        if len(fails):
            logger.log("★ 存在未达标组合 —— 按预注册必须撤回摘要中「两种不同泄漏机制」的主张")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
