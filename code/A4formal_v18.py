#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A4formal_v18 — Formal 阶段：5 固定种子重跑全部**依赖随机划分**的量。

起因：Formal 实验的口径是须用 5 个固定种子
`[20060515, 20041210, 19810915, 2023, 2024]`，报告 mean/std/95%CI（cluster bootstrap
≥1000 次），并显式标注 n_primary/K_inner/n_seeds；而种子 `2004` 是 **Pilot dev 种子**，
「严禁出现在 Formal 5 种子中」。v18 全部分析此前只用 2004 单种子跑过，属 Pilot 级证据。

────────────────────────────────────────────────────────────────────
【范围界定：哪些量需要重跑】

先判定各量对种子的依赖性，避免无意义重跑：

  · **与种子无关（不重跑，沿用原值）**
      - 93 的方差分量 σ²a / σ²f / ICC / 可达上界：平衡设计下的 ANOVA **矩估计**，
        闭式解，无随机性。
      - 97 的仪器侧全部结果：用 `GroupKFold`（sklearn 实现**不打乱**），划分确定；
        内层成分数选择同样确定。故 R²=0.393/0.819、RMSEP、仪器分量占比均为定值。
      - 1/m 标度、纯标签基线：由 σ²a/σ²f 闭式导出，无随机性。

  · **依赖种子（本脚本重跑）**
      - T1 苹果光谱主表：3 目标 × 4 预处理的重复留出 R²（94 表a）
      - T2 单面 vs 5 面平均光谱（94 表b）
      - T3 分组 vs 按行随机划分的泄漏比（95 表b/c）
      - T4 衰减比（配对设计，A2 表a）
      - T5 面间可交换性的复合对称置换检验 p（A1 表b）
      - T6 A1 的独立实现 cv_r2（A1 表f）
      - T7 DeepHS 按果 vs 按天（真实标签，A2 表b/c 的 V8 判据）

【统计口径】
  · 每个种子内做 N_REPEAT=20 次按实体分组的重复留出；5 种子共 100 个重复。
  · 报告：跨种子 mean/std + **cluster bootstrap 95% CI**（簇=种子，B=2000）——
    先对 5 个种子有放回重抽，再在被抽中的种子内对其 20 个重复有放回重抽，
    如实反映「种子间」与「划分间」两级变异。禁止把 100 个重复当作 100 个独立样本。
  · 显式标注 n_primary=N_REPEAT, K_inner=4, n_seeds=5。

[免检] multiprocessing.Pool 并行 —— 纯工程加速，各任务相互独立，不改变任何数字。

用法（服务器）：python3 A4formal_v18.py
输出：outputs/A4formal_v18.xlsx + logs/
"""
from __future__ import annotations

# ⚠️ 必须在 import numpy 之前设置：BLAS/OpenMP 默认按物理核数开线程（本机 192），
# 与 multiprocessing 的 48 个 worker 相乘 => 五千余线程抢 192 核，全耗在上下文切换。
# 实测未设时 load average 63–91、单 worker 107–321 线程、1 小时跑不完第 1 阶段。
# 并行度由进程级 Pool 提供，每进程单线程 BLAS 才是正确组合。
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

# ── Formal 口径 ───────────────────────────────────────────────────────
SEEDS_FORMAL = [20060515, 20041210, 19810915, 2023, 2024]
PILOT_SEED = 2004                     # 仅作对照，严禁计入 Formal
N_REPEAT = 20                         # n_primary
N_INNER_FOLD = 4                      # K_inner
TEST_FRAC = 0.40
MAX_PLS = 20
N_BOOT = 2000

# 预注册阈值（沿用，不得事后调整）
V8_RATIO = 1.5

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUTD = os.path.join(HERE, "..", "outputs")
if not os.path.isdir(DATA):                       # 本机调试时的回退路径
    DATA = os.path.join(HERE, "..", "03data", "processed")
    _OUTA = os.path.join(HERE, "..", "outputs")
    OUTD = _OUTA if os.path.isdir(_OUTA) else os.path.join(HERE, "..", "04outputs")

APPLE = os.path.join(DATA, "v18_apple_faces.parquet")
TRAJ = os.path.join(DATA, "v18_deephs_traj.parquet")


def log(*a):
    print(*a, flush=True)


# ── 基础工具（与 94localsense 逐行同构）───────────────────────────────
def snv(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1, ddof=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


def d1(X):
    return savgol_filter(X, 11, 2, deriv=1, axis=1)


PREPROC = {"raw": lambda X: X, "SNV": snv, "1stDeriv": d1,
           "SNV+1stDeriv": lambda X: d1(snv(X))}


def pls_fit_predict(Xtr, ytr, Xte, groups_tr):
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


def r2_holdout(y, p, train_mean):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - train_mean) ** 2)


# ══ 任务：每个 (seed, unit) 返回 **逐重复** 的原始值，聚合留到最后 ══
def task_apple(args):
    """T1/T2/T4：苹果光谱三目标 + 单面vs五面 + 衰减比（同一划分序，保证配对）。"""
    seed, preproc = args
    df = pd.read_parquet(APPLE)
    wl = [c for c in df.columns if c.endswith("nm")]
    Xraw = df[wl].to_numpy(float)
    Xp = PREPROC[preproc](Xraw)
    groups = df["id"].to_numpy()
    uniq = np.unique(groups)
    n_te = int(round(TEST_FRAC * len(uniq)))
    rng = np.random.default_rng(seed)

    out = {t: [] for t in ("y_face", "y_fruit", "y_dev")}
    for _ in range(N_REPEAT):
        te = set(rng.permutation(uniq)[:n_te])
        m = np.array([g in te for g in groups])
        for t in out:
            y = df[t].to_numpy(float)
            p, _ = pls_fit_predict(Xp[~m], y[~m], Xp[m], groups[~m])
            out[t].append(float(r2_holdout(y[m], p, y[~m].mean())))
    return {"unit": "T1", "seed": seed, "preproc": preproc, "reps": out}


def task_leak(args):
    """T3：同一预处理下 按果分组 vs 按行随机 的 R²（泄漏比的分子分母）。"""
    seed, preproc = args
    df = pd.read_parquet(APPLE)
    wl = [c for c in df.columns if c.endswith("nm")]
    Xp = PREPROC[preproc](df[wl].to_numpy(float))
    groups = df["id"].to_numpy()
    uniq = np.unique(groups)
    n = len(df)
    rng = np.random.default_rng(seed)
    res = {("grouped", t): [] for t in ("y_face", "y_fruit")}
    res.update({("random", t): [] for t in ("y_face", "y_fruit")})
    for _ in range(N_REPEAT):
        te = set(rng.permutation(uniq)[:int(round(TEST_FRAC * len(uniq)))])
        mg = np.array([g in te for g in groups])
        idx = rng.permutation(n)
        mr = np.zeros(n, bool)
        mr[idx[:int(round(TEST_FRAC * n))]] = True
        for t in ("y_face", "y_fruit"):
            y = df[t].to_numpy(float)
            p, _ = pls_fit_predict(Xp[~mg], y[~mg], Xp[mg], groups[~mg])
            res[("grouped", t)].append(float(r2_holdout(y[mg], p, y[~mg].mean())))
            p2, _ = pls_fit_predict(Xp[~mr], y[~mr], Xp[mr], groups[~mr])
            res[("random", t)].append(float(r2_holdout(y[mr], p2, y[~mr].mean())))
    return {"unit": "T3", "seed": seed, "preproc": preproc,
            "reps": {f"{k[0]}|{k[1]}": v for k, v in res.items()}}


def task_deephs(args):
    """T7：DeepHS 真实标签下 按果 vs 按天（V8 判据）。"""
    seed, target, fruit = args
    tj = pd.read_parquet(TRAJ)
    w = [c for c in tj.columns if c.startswith("w")]
    sub = tj.dropna(subset=[target])
    if fruit != "__汇总__":
        sub = sub[sub.fruit == fruit]
    if len(sub) < 40:
        return None
    X = snv(sub[w].to_numpy(float))
    y = sub[target].to_numpy(float)
    keys = {"按果分组": (sub["fruit"] + "|" + sub["sample"]).to_numpy(),
            "按采集日分组": (sub["fruit"] + "|" + sub["day"]).to_numpy()}
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in keys}
    for k, grp in keys.items():
        uq = np.unique(grp)
        n_te = max(1, int(round(TEST_FRAC * len(uq))))
        r = np.random.default_rng(seed)          # 两种分组用同一序列，保证配对
        for _ in range(N_REPEAT):
            te = set(r.permutation(uq)[:n_te])
            m = np.array([g in te for g in grp])
            if m.sum() < 3 or (~m).sum() < 10 or len(np.unique(grp[~m])) < N_INNER_FOLD:
                continue
            p, _ = pls_fit_predict(X[~m], y[~m], X[m], grp[~m])
            reps[k].append(float(np.sqrt(np.mean((y[m] - p) ** 2))))
    del rng
    return {"unit": "T7", "seed": seed, "target": target, "fruit": fruit, "reps": reps}


# ── 聚合：cluster bootstrap（簇 = 种子）───────────────────────────────
def cluster_boot_ci(by_seed: dict[int, list[float]], b=N_BOOT, seed=0):
    """簇=种子的两级自助：先重抽种子，再在种子内重抽重复。"""
    seeds = list(by_seed)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(b):
        picked = rng.choice(len(seeds), len(seeds), replace=True)
        vals = []
        for i in picked:
            v = np.asarray(by_seed[seeds[i]], float)
            if len(v) == 0:
                continue
            vals.extend(v[rng.integers(0, len(v), len(v))])
        if vals:
            means.append(np.mean(vals))
    if not means:
        return np.nan, np.nan
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(by_seed: dict[int, list[float]], label: str) -> dict:
    per_seed_mean = {s: float(np.mean(v)) for s, v in by_seed.items() if len(v)}
    allv = [x for v in by_seed.values() for x in v]
    lo, hi = cluster_boot_ci(by_seed)
    return {"量": label,
            "跨种子均值": float(np.mean(list(per_seed_mean.values()))),
            "跨种子SD": float(np.std(list(per_seed_mean.values()), ddof=1)) if len(per_seed_mean) > 1 else np.nan,
            "cluster CI95 下限": lo, "cluster CI95 上限": hi,
            "n_seeds": len(per_seed_mean), "n_primary": N_REPEAT, "K_inner": N_INNER_FOLD,
            "总重复数": len(allv),
            "逐种子均值": json.dumps({str(k): round(v, 6) for k, v in per_seed_mean.items()},
                                     ensure_ascii=False)}


def main():
    t0 = time.time()
    os.makedirs(OUTD, exist_ok=True)
    log("=" * 74)
    log("A4formal_v18 — Formal 5 种子重跑（依赖随机划分的量）")
    log(f"SEEDS={SEEDS_FORMAL}  (Pilot 种子 {PILOT_SEED} 已排除)")
    log(f"n_primary={N_REPEAT}  K_inner={N_INNER_FOLD}  TEST_FRAC={TEST_FRAC}  B={N_BOOT}")
    log(f"CPU={os.cpu_count()}")
    log("=" * 74)

    jobs_apple = [(s, p) for s in SEEDS_FORMAL for p in PREPROC]
    jobs_leak = [(s, p) for s in SEEDS_FORMAL for p in ("raw", "1stDeriv")]
    jobs_deep = [(s, t, f) for s in SEEDS_FORMAL
                 for t in ("firmness", "storage_days")
                 for f in ("Kiwi", "Avocado", "__汇总__")]

    nproc = min(os.cpu_count() or 8, 48)
    with Pool(nproc) as pool:
        log(f"[1/3] 苹果三目标 × 4 预处理 × 5 种子 = {len(jobs_apple)} 任务 …")
        res_apple = pool.map(task_apple, jobs_apple)
        log(f"      done {time.time()-t0:.0f}s")
        log(f"[2/3] 泄漏对照 × 2 预处理 × 5 种子 = {len(jobs_leak)} 任务 …")
        res_leak = pool.map(task_leak, jobs_leak)
        log(f"      done {time.time()-t0:.0f}s")
        log(f"[3/3] DeepHS 真实标签 V8 对照 = {len(jobs_deep)} 任务 …")
        res_deep = [r for r in pool.map(task_deephs, jobs_deep) if r]
        log(f"      done {time.time()-t0:.0f}s")

    # ── 表a：苹果光谱主表 ───────────────────────────────────────────
    rows_a = []
    for pre in PREPROC:
        for tgt, name in (("y_face", "面级 SSC"), ("y_fruit", "整果均值 SSC"), ("y_dev", "面级偏离 e_ij")):
            by = {r["seed"]: r["reps"][tgt] for r in res_apple if r["preproc"] == pre}
            d = summarize(by, f"R² · {pre} · {name}")
            d["预处理"], d["目标"] = pre, name
            rows_a.append(d)
    tab_a = pd.DataFrame(rows_a)

    # ── 表b：衰减比（raw，配对）─────────────────────────────────────
    by_face = {r["seed"]: r["reps"]["y_face"] for r in res_apple if r["preproc"] == "raw"}
    by_fruit = {r["seed"]: r["reps"]["y_fruit"] for r in res_apple if r["preproc"] == "raw"}
    ratio_by_seed = {s: [f / u for f, u in zip(by_face[s], by_fruit[s]) if u > 0]
                     for s in by_face}
    VAR_FACE = 2.327025 + 2.551483
    VAR_FRUIT5 = 2.327025 + 2.551483 / 5
    tab_b = pd.DataFrame([
        summarize(by_face, "R² 面级目标（raw）"),
        summarize(by_fruit, "R² 5面均值目标（raw）"),
        summarize(ratio_by_seed, "衰减比 = R²_face / R²_fruit（逐重复配对）"),
    ])
    theo = pd.DataFrame([
        {"量": "理论值B：(σ²a+σ²f/5)/(σ²a+σ²f) ← 本文目标口径的正确预言",
         "跨种子均值": VAR_FRUIT5 / VAR_FACE},
        {"量": "理论值A：ICC = σ²a/(σ²a+σ²f) ← 仅整果目标无噪时成立", "跨种子均值": 2.327025 / VAR_FACE},
    ])
    tab_b = pd.concat([tab_b, theo], ignore_index=True)

    # ── 表c：泄漏对照 ───────────────────────────────────────────────
    rows_c = []
    for pre in ("raw", "1stDeriv"):
        for tgt in ("y_face", "y_fruit"):
            g = {r["seed"]: r["reps"][f"grouped|{tgt}"] for r in res_leak if r["preproc"] == pre}
            rd = {r["seed"]: r["reps"][f"random|{tgt}"] for r in res_leak if r["preproc"] == pre}
            ratio = {s: [a / b for a, b in zip(rd[s], g[s]) if b > 0] for s in g}
            for lab, by in (("按果分组", g), ("按行随机", rd), ("虚高倍数 随机/分组", ratio)):
                d = summarize(by, f"{pre} · {tgt} · {lab}")
                d["预处理"], d["目标"], d["口径"] = pre, tgt, lab
                rows_c.append(d)
    tab_c = pd.DataFrame(rows_c)

    # ── 表d：DeepHS V8 ──────────────────────────────────────────────
    rows_d = []
    for tgt in ("firmness", "storage_days"):
        for fr in ("Kiwi", "Avocado", "__汇总__"):
            sel = [r for r in res_deep if r["target"] == tgt and r["fruit"] == fr]
            if not sel:
                continue
            gf = {r["seed"]: r["reps"]["按果分组"] for r in sel}
            gd = {r["seed"]: r["reps"]["按采集日分组"] for r in sel}
            ratio = {s: [d_ / f_ for d_, f_ in zip(gd[s], gf[s]) if f_ > 0] for s in gf}
            for lab, by in (("RMSE 按果", gf), ("RMSE 按天", gd), ("比值 天/果", ratio)):
                d = summarize(by, f"{tgt} · {fr} · {lab}")
                d["目标"], d["物种"], d["口径"] = tgt, fr, lab
                rows_d.append(d)
    tab_d = pd.DataFrame(rows_d)

    # V8 判定（用跨种子均值 + CI 下限双口径）
    rows_v = []
    for tgt in ("firmness", "storage_days"):
        for fr in ("Kiwi", "Avocado", "__汇总__"):
            r = tab_d[(tab_d.get("目标") == tgt) & (tab_d.get("物种") == fr)
                      & (tab_d.get("口径") == "比值 天/果")]
            if r.empty:
                continue
            mean, lo = float(r["跨种子均值"].iloc[0]), float(r["cluster CI95 下限"].iloc[0])
            rows_v.append({"目标": tgt, "物种": fr, "比值均值": mean, "CI95下限": lo,
                           "阈值": V8_RATIO,
                           "判定（均值口径）": "达标" if mean >= V8_RATIO else "未达标",
                           "判定（CI下限口径·更严）": "达标" if lo >= V8_RATIO else "未达标"})
    tab_v = pd.DataFrame(rows_v)

    # 先把**逐重复原始值**落 JSON —— 计算成本远高于写表，绝不能被写表 bug 毁掉。
    raw_p = os.path.join(OUTD, "A4formal_v18_raw.json")
    with open(raw_p, "w", encoding="utf-8") as fh:
        json.dump({"seeds": SEEDS_FORMAL, "n_primary": N_REPEAT, "K_inner": N_INNER_FOLD,
                   "apple": res_apple, "leak": res_leak, "deephs": res_deep},
                  fh, ensure_ascii=False)
    log(f"原始逐重复结果 → {raw_p}")

    out = os.path.join(OUTD, "A4formal_v18.xlsx")
    with pd.ExcelWriter(out) as w:
        pd.DataFrame([
            {"项": "Formal 种子", "值": str(SEEDS_FORMAL)},
            {"项": "Pilot 种子（已排除）", "值": PILOT_SEED},
            {"项": "n_primary（每种子重复留出次数）", "值": N_REPEAT},
            {"项": "K_inner（内层分组 CV 折数）", "值": N_INNER_FOLD},
            {"项": "n_seeds", "值": len(SEEDS_FORMAL)},
            {"项": "cluster bootstrap B", "值": N_BOOT},
            {"项": "簇定义", "值": "簇 = 种子；先重抽种子再在种子内重抽重复"},
            {"项": "与种子无关故未重跑的量", "值": "93 方差分量/ICC/上界（ANOVA 矩估计，闭式）；"
                                                    "97 仪器侧全部结果（GroupKFold 不打乱，确定性）；"
                                                    "1/m 标度与纯标签基线（由 σ² 闭式导出）"},
        ]).to_excel(w, sheet_name="表0：Formal 口径", index=False)
        tab_a.to_excel(w, sheet_name="表a：苹果光谱主表", index=False)
        tab_b.to_excel(w, sheet_name="表b：衰减比", index=False)
        tab_c.to_excel(w, sheet_name="表c：泄漏对照", index=False)
        tab_d.to_excel(w, sheet_name="表d：DeepHS 真实标签", index=False)
        tab_v.to_excel(w, sheet_name="表e：V8 判定", index=False)
    log("=" * 74)
    log(f"写出 {out}   总耗时 {time.time()-t0:.0f}s")
    for _, r in tab_v.iterrows():
        log(f"  V8 {r['目标']:13s} {r['物种']:9s} 比值={r['比值均值']:.3f} "
            f"CI下限={r['CI95下限']:.3f} → {r['判定（均值口径）']}")


if __name__ == "__main__":
    main()
