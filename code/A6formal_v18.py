#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A6formal_v18 — Formal 5 种子补跑 A4 未覆盖的四组依赖随机划分的量。

A4formal 覆盖了苹果三目标主表、泄漏对照、DeepHS 真实标签 V8 三项；稿件里还有四组
数字仍停留在 pilot 种子 2004，须改用 Formal 5 种子复跑：

  T8  苹果波段子集建模（全谱 / 无饱和波段 / 860-1001 / 900-1001 nm）
      —— 稿件 4.3「全谱 0.097、仅用无饱和波段 0.069、900-1001 nm 0.012」
      刻意沿用 A1consolidate 的评估器（重复 15 次、成分数网格 2:24:2、内层 GroupKFold(4)），
      因为 4.3 的论证前提正是「这是与 A4 不同的另一套实现」，统一参数会毁掉该论证。
  T9  五面平均光谱 → 整果均值 SSC —— 稿件 4.3 的 0.097 之外的对照口径（94 表b 第 2 行）
      用 A4formal 的评估器（重复 20 次、成分数 1..20、内层 GroupKFold(4)），
      使之与 A4 已有的「单面 → 整果均值」构成同评估器配对。
  T10 猕猴桃 012 强模型 DM / SSC —— 稿件 4.3「同一套代码与协议在猕猴桃数据上得 DM / SSC 的 R²」
      沿用 A1consolidate 评估器与 nrep=6。
  T11 DeepHS 成像日序 按果 vs 按天 —— 稿件 3.5 中**已声明撤回**、但仍逐个报告的那组观察。
      来源是 A1consolidate 表i（`cv_r2`，含随机留出），**不是** 99deephskill 表c/表d。
      99deephskill 用 GroupKFold(n_splits=5) 无 shuffle，确实与种子无关；两者别混。
      沿用 A1consolidate 评估器与 nrep=6、成分数网格 2:20:2。

不重跑的量及理由（已逐一核对，判据是脚本内是否调用 rng / shuffle）：
  · 97instrbudget：分组 5 折 GroupKFold，无 shuffle → 划分确定，与种子无关。
  · 99deephskill 表c/表d：同上。
  · 93sscceiling 的方差分量 / ICC / 可达上界、95 表e 的纯标签基线：闭式矩估计 → 与种子无关。
  · 95 表a 的光谱质量诊断（Pearson r、饱和率）、96satdiag 全表：无划分，与种子无关。

[免检] 本脚本不引入任何新方法，只是把已登记方法（M4/M9/M10）的运行种子由 pilot 换成
Formal 5 种子并按 M10 的 cluster bootstrap 聚合；覆盖率核查本身登记为 M14。

运行方式:
    python3 code/A6formal_v18.py

输出文件:
    outputs/A6formal_v18.xlsx        — 四组量的 Formal 5 种子结果
    outputs/A6formal_v18_raw.json    — 逐重复原始值（先落盘再写 Excel）
"""
from __future__ import annotations

# 必须在 import numpy 之前锁 BLAS 线程，否则 multiprocessing 下线程超订（见记忆 blas-oversubscription）
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

SEEDS_FORMAL = [20060515, 20041210, 19810915, 2023, 2024]
TEST_FRAC = 0.40
N_BOOT = 2000

# A1consolidate 评估器（T8 / T10 沿用）
A1_NREP, A1_MAXC, A1_INNER = 15, 24, 4
# A4formal 评估器（T9 沿用）
A4_NREP, A4_MAXC, A4_INNER = 20, 20, 4

HERE = _os.path.dirname(_os.path.abspath(__file__))
DATA = _os.path.join(HERE, "..", "03data", "processed")
if not _os.path.isdir(DATA):                       # 服务器布局
    DATA = _os.path.join(HERE, "..", "data")
_OUTA = _os.path.join(HERE, "..", "outputs")
OUT = _OUTA if _os.path.isdir(_OUTA) else _os.path.join(HERE, "..", "04outputs")
if not _os.path.isdir(OUT):
    OUT = _os.path.join(HERE, "..", "outputs")


def d1(X):
    return savgol_filter(X, 11, 2, deriv=1, axis=1)


def cv_a1(X, y, groups, seed, nrep=A1_NREP, max_comp=A1_MAXC):
    """A1consolidate 口径：成分数网格 2:max_comp:2，内层 GroupKFold(4)。"""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    r2s, rms = [], []
    for _ in range(nrep):
        te = set(rng.permutation(uq)[: int(round(TEST_FRAC * len(uq)))])
        m = np.array([g in te for g in groups])
        Xtr, ytr, gtr = X[~m], y[~m], groups[~m]
        inner = list(GroupKFold(n_splits=A1_INNER).split(Xtr, ytr, gtr))
        best, berr = 2, np.inf
        for c in range(2, min(max_comp, Xtr.shape[1]) + 1, 2):
            e = [np.mean((PLSRegression(c).fit(Xtr[a], ytr[a]).predict(Xtr[b]).ravel()
                          - ytr[b]) ** 2) for a, b in inner]
            if np.mean(e) < berr - 1e-9:
                berr, best = float(np.mean(e)), c
        p = PLSRegression(best).fit(Xtr, ytr).predict(X[m]).ravel()
        r2s.append(float(1 - np.sum((y[m] - p) ** 2) / np.sum((y[m] - ytr.mean()) ** 2)))
        rms.append(float(np.sqrt(np.mean((y[m] - p) ** 2))))
    return r2s, rms


def cv_a4(X, y, groups, seed, nrep=A4_NREP, max_comp=A4_MAXC):
    """A4formal 口径：成分数逐一搜 1..max_comp，内层 GroupKFold(4)，SST 用训练折均值。"""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    r2s, rms = [], []
    for _ in range(nrep):
        te = set(rng.permutation(uq)[: int(round(TEST_FRAC * len(uq)))])
        m = np.array([g in te for g in groups])
        Xtr, ytr, gtr = X[~m], y[~m], groups[~m]
        ncmax = min(max_comp, Xtr.shape[1], max(2, Xtr.shape[0] // 10))
        inner = list(GroupKFold(n_splits=min(A4_INNER, len(np.unique(gtr))))
                     .split(Xtr, ytr, gtr))
        best, berr = 1, np.inf
        for c in range(1, ncmax + 1):
            e = [np.mean((PLSRegression(c, scale=False).fit(Xtr[a], ytr[a])
                          .predict(Xtr[b]).ravel() - ytr[b]) ** 2) for a, b in inner]
            if np.mean(e) < berr - 1e-9:
                berr, best = float(np.mean(e)), c
        p = PLSRegression(best, scale=False).fit(Xtr, ytr).predict(X[m]).ravel()
        r2s.append(float(1 - np.sum((y[m] - p) ** 2) / np.sum((y[m] - ytr.mean()) ** 2)))
        rms.append(float(np.sqrt(np.mean((y[m] - p) ** 2))))
    return r2s, rms


def boot_mean(by, b=N_BOOT, seed=0):
    """簇 = 种子的两级 cluster bootstrap（M10）。"""
    ks = [k for k in by if len(by[k])]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        v = []
        for i in rng.choice(len(ks), len(ks), replace=True):
            a = np.asarray(by[ks[i]], float)
            v.extend(a[rng.integers(0, len(a), len(a))])
        vals.append(np.mean(v))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(by, label, extra=None):
    per = {k: float(np.mean(v)) for k, v in by.items() if len(v)}
    lo, hi = boot_mean(by)
    d = {"量": label, "跨种子均值": float(np.mean(list(per.values()))),
         "跨种子SD": float(np.std(list(per.values()), ddof=1)),
         "cluster CI95 下限": lo, "cluster CI95 上限": hi,
         "n_seeds": len(per),
         "逐种子": json.dumps({str(k): round(v, 4) for k, v in per.items()})}
    if extra:
        d.update(extra)
    return d


# ── 任务函数（顶层定义，供 multiprocessing pickle）────────────────────────
def task_band(a):
    tag, mask, seed = a
    ap = pd.read_parquet(_os.path.join(DATA, "v18_apple_faces.parquet"))
    wc = [c for c in ap.columns if c.endswith("nm")]
    wl = np.array([float(c[:-2]) for c in wc])
    X = ap[wc].values.astype(float)
    lo = (X < 0.01).mean(0) * 100
    hi = (X > 0.99).mean(0) * 100
    sel = {"全谱 346 波段": np.ones(len(wl), bool),
           "仅无饱和波段": (lo + hi) <= 1,
           "860–1001 nm": wl >= 860,
           "900–1001 nm（糖倍频带）": wl >= 900}[tag]
    r2s, _ = cv_a1(X[:, sel], ap["y_fruit"].values.astype(float), ap["id"].values, seed)
    return {"task": "T8", "tag": tag, "n_band": int(sel.sum()), "seed": seed, "reps": r2s}


def task_meanspec(seed):
    ap = pd.read_parquet(_os.path.join(DATA, "v18_apple_faces.parquet"))
    wc = [c for c in ap.columns if c.endswith("nm")]
    ids = ap["id"].values
    uq = np.unique(ids)
    Xm = np.vstack([ap.loc[ids == f, wc].values.astype(float).mean(0) for f in uq])
    ym = np.array([ap.loc[ids == f, "ssc"].mean() for f in uq], float)
    r2s, _ = cv_a4(Xm, ym, uq, seed)
    return {"task": "T9", "tag": "5 面光谱平均 → 整果均值 SSC（raw）", "seed": seed, "reps": r2s}


def task_kiwi(a):
    tgt, seed = a
    kd = pd.read_parquet(_os.path.join(DATA, "v18_kiwi_instr.parquet")).dropna(subset=["SSC", "DM"])
    kw = [c for c in kd.columns if c.startswith("X")]
    V = kd[kw].values.astype(float)
    ok = ~np.isnan(V).any(0)
    kw = [c for c, b in zip(kw, ok) if b]
    kwl = np.array([float(c[1:]) for c in kw])
    KX = d1(kd[kw].values.astype(float)[:, kwl >= 700])
    r2s, rms = cv_a1(KX, kd[tgt].values.astype(float), kd["sample_id"].values, seed, nrep=6)
    return {"task": "T10", "tag": tgt, "seed": seed, "reps": r2s, "rmse": rms}


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, ddof=1, keepdims=True) + 1e-12)


DAY_SETTINGS = ["全体·按果分组", "全体·按采集日分组", "Kiwi·按果分组",
                "Kiwi·按采集日分组", "Avocado·按果分组", "Avocado·按采集日分组"]


def task_daysplit(a):
    """T11：DeepHS 成像日序 按果 vs 按天（稿件 3.5 中**已被撤回**的那组观察）。

    这组数字虽已在正文声明撤回，但仍作为「协议陷阱」的实证被逐个报告，故同样须走
    Formal 5 种子。沿用 A1consolidate 表i 的评估器（nrep=6，成分数网格 2:20:2）。
    """
    tag, seed = a
    tj = pd.read_parquet(_os.path.join(DATA, "v18_deephs_traj.parquet"))
    twc = [c for c in tj.columns if c.startswith("w")]
    TX = snv(tj[twc].to_numpy(float))
    ty = tj["day_idx"].to_numpy(float)
    fk = (tj["fruit"] + "|" + tj["sample"]).to_numpy()
    dk = (tj["fruit"] + "|" + tj["day"]).to_numpy()
    sp, how = tag.split("·")
    mask = np.ones(len(tj), bool) if sp == "全体" else (tj["fruit"] == sp).values
    if how == "按果分组":
        grp = fk
    else:                       # 分物种时 fruit 恒定，故 fruit|day 与 day 等价（沿用原实现）
        grp = dk if sp == "全体" else tj["day"].to_numpy()
    r2s, rms = cv_a1(TX[mask], ty[mask], grp[mask], seed, nrep=6, max_comp=20)
    return {"task": "T11", "tag": tag, "n": int(mask.sum()), "seed": seed,
            "reps": r2s, "rmse": rms}


def main() -> None:
    t0 = time.time()
    print("=" * 74)
    print(f"A6formal_v18 — Formal 5 种子补跑  SEEDS={SEEDS_FORMAL}")
    print(f"T8/T10 用 A1 评估器(nrep={A1_NREP}/6, 网格 2:{A1_MAXC}:2, inner={A1_INNER})；"
          f"T9 用 A4 评估器(nrep={A4_NREP}, 网格 1..{A4_MAXC}, inner={A4_INNER})")
    print(f"DATA={_os.path.realpath(DATA)}  CPU={mp.cpu_count()}")
    print("=" * 74)

    bands = ["全谱 346 波段", "仅无饱和波段", "860–1001 nm", "900–1001 nm（糖倍频带）"]
    nw = min(20, mp.cpu_count() - 2)
    with mp.Pool(nw) as pool:
        print(f"[1/3] T8 苹果波段子集 {len(bands)}×5 = {len(bands)*5} 任务 …", flush=True)
        r8 = pool.map(task_band, [(b, None, s) for b in bands for s in SEEDS_FORMAL])
        print(f"      done {time.time()-t0:.0f}s", flush=True)
        print("[2/3] T9 五面平均光谱 5 任务 …", flush=True)
        r9 = pool.map(task_meanspec, SEEDS_FORMAL)
        print(f"      done {time.time()-t0:.0f}s", flush=True)
        print("[3/4] T10 猕猴桃强模型 2×5 = 10 任务 …", flush=True)
        r10 = pool.map(task_kiwi, [(t, s) for t in ("DM", "SSC") for s in SEEDS_FORMAL])
        print(f"      done {time.time()-t0:.0f}s", flush=True)
        print(f"[4/4] T11 DeepHS 成像日序 {len(DAY_SETTINGS)}×5 = "
              f"{len(DAY_SETTINGS)*5} 任务 …", flush=True)
        r11 = pool.map(task_daysplit, [(t, s) for t in DAY_SETTINGS for s in SEEDS_FORMAL])
        print(f"      done {time.time()-t0:.0f}s", flush=True)

    raw = {"seeds": SEEDS_FORMAL, "A1_nrep": A1_NREP, "A4_nrep": A4_NREP,
           "K_inner": A1_INNER, "band": r8, "meanspec": r9, "kiwi": r10, "day": r11}
    rp = _os.path.join(OUT, "A6formal_v18_raw.json")
    json.dump(raw, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"原始逐重复结果 → {rp}", flush=True)

    rows8 = [summarize({r["seed"]: r["reps"] for r in r8 if r["tag"] == b},
                       f"苹果 · {b} → 整果均值 SSC",
                       {"波段数": [r["n_band"] for r in r8 if r["tag"] == b][0],
                        "n_primary": A1_NREP, "K_inner": A1_INNER}) for b in bands]
    rows9 = [summarize({r["seed"]: r["reps"] for r in r9}, r9[0]["tag"],
                       {"n_primary": A4_NREP, "K_inner": A4_INNER})]
    rows10 = [summarize({r["seed"]: r["reps"] for r in r10 if r["tag"] == t},
                        f"猕猴桃 012 · {t} · R²（一阶导 700–1065nm，按 sample_id 分组）",
                        {"n_primary": 6, "K_inner": A1_INNER}) for t in ("DM", "SSC")]
    rows10 += [summarize({r["seed"]: r["rmse"] for r in r10 if r["tag"] == t},
                         f"猕猴桃 012 · {t} · RMSEP", {"n_primary": 6, "K_inner": A1_INNER})
               for t in ("DM", "SSC")]

    rows11 = []
    for t in DAY_SETTINGS:
        sel = [r for r in r11 if r["tag"] == t]
        rows11.append(summarize({r["seed"]: r["reps"] for r in sel}, f"日序 · {t} · R²",
                                {"n谱": sel[0]["n"], "n_primary": 6, "K_inner": A1_INNER}))
        rows11.append(summarize({r["seed"]: r["rmse"] for r in sel}, f"日序 · {t} · RMSE",
                                {"n谱": sel[0]["n"], "n_primary": 6, "K_inner": A1_INNER}))

    sheets = {0: ("T8 苹果波段子集", pd.DataFrame(rows8)),
              1: ("T9 五面平均光谱", pd.DataFrame(rows9)),
              2: ("T10 猕猴桃强模型", pd.DataFrame(rows10)),
              3: ("T11 DeepHS 日序分组", pd.DataFrame(rows11))}
    sys.path.insert(0, HERE)
    try:
        from export_utils import write_script_workbook
        out = write_script_workbook(__file__, sheets)
    except Exception as e:                                    # 服务器无 export_utils 时兜底
        out = _os.path.join(OUT, "A6formal_v18.xlsx")
        print(f"(export_utils 不可用：{e}；改用裸 ExcelWriter)")
        with pd.ExcelWriter(out) as w:
            for _, (nm, dfr) in sorted(sheets.items()):
                dfr.to_excel(w, sheet_name=nm, index=False)

    print("=" * 74)
    for r in rows8 + rows9 + rows10 + rows11:
        print(f"  {r['量'][:46]:46s} {r['跨种子均值']:+.4f} "
              f"CI95[{r['cluster CI95 下限']:+.4f}, {r['cluster CI95 上限']:+.4f}]")
    print(f"写出 {out}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
