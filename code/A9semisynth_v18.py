#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A9semisynth_v18 — 预注册 V9：半合成参考噪声退化实验。

**判据在本脚本编写之前已锁定**，见 `docs/0NSTATISTICAL_PROTOCOL.md` 附录 G。
本脚本只执行，不得修改判据；结果无论正负一律如实落盘。

问题（预投稿评审中被指出）：本文核心主张是「参考值变异为任何只能感知整果
水平信息的模型设定 R² 上界」，但苹果数据的光谱侧 R²≈0.11，**该上界在本文数据上从未被逼近
过**，因而"可证伪"这一卖点没有被真正触发。

做法：换到光谱侧确实可用的猕猴桃数据（DM 定标 R²≈0.83、SSC≈0.86），按苹果实测的方差比
向标签**注入**参考噪声，让模型在带噪标签上**重新训练**，看性能是否沿无自由参数的预言退化、
并被上界卡住。

  预言（无自由参数）  R²_pred(m) = R²_clean · σ_y² / (σ_y² + σ_f²/m)
  上界（式 eq:ceiling）        σ_a² / (σ_a² + σ_f²/m)

  P1 硬证伪：每个 m，实测 R² 的 cluster CI 下限不得高于上界
  P2 跟踪度：|R²_obs/R²_pred − 1| 跨 m 的中位数 ≤ 15%
  P3 绑定性：至少一个 m ≤ 5 使 R²_obs ≥ 0.8 · R²_pred(m)

关键实现细节（都会改变结论，逐条说明）：
  · 噪声按**果**注入，同一果的所有光谱共享同一个带噪标签 —— 参考测定是逐果做的，不是逐谱。
  · 噪声在**每个重复内重新抽样**，且训练/测试共用同一实现 —— 模拟"这一批标签就是这样"。
  · 成分数选择也用带噪标签 —— 真实实验者手里只有带噪参考值，不可能用干净标签选模型。
  · R² 对**带噪标签**计算 —— 这才是实验者实际会报告的数，也正是理论所约束的量。
    另附对干净标签的 R² 作诊断列（不参与判据）。

运行方式:
    python3 code/A9semisynth_v18.py

输出文件:
    outputs/A9semisynth_v18.xlsx      — 表a 退化轨迹；表b 预注册判定
    outputs/A9semisynth_v18_raw.json  — 逐重复原始值
"""
from __future__ import annotations

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

# ── 预注册常量（附录 G，不得在看到结果后修改）────────────────────────────
SEEDS_FORMAL = [20060515, 20041210, 19810915, 2023, 2024]
M_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 20, 0]   # 0 代表 m=∞；V9b 加密网格覆盖全部 m_req
SIGMA_A2_APPLE, SIGMA_F2_APPLE = 2.327025, 2.551483      # 93sscceiling 表c，闭式
RATIO = SIGMA_F2_APPLE / SIGMA_A2_APPLE                   # 1.096457
NREP, MAXC, INNER, TEST_FRAC = 6, 24, 4, 0.40             # 与 A6 的 T10 一致
N_BOOT = 2000
P2_TOL, P3_FRAC = 0.15, 0.80
# P4 事前指定目标（附录 H.3 锁定）：(目标, R2_t, 是否盲检)
P4_TARGETS = [("DM", 0.70, False), ("DM", 0.75, True), ("DM", 0.78, True),
              ("SSC", 0.70, False), ("SSC", 0.75, True), ("SSC", 0.80, True)]

HERE = _os.path.dirname(_os.path.abspath(__file__))
DATA = _os.path.join(HERE, "..", "03data", "processed")
if not _os.path.isdir(DATA):
    DATA = _os.path.join(HERE, "..", "data")
_OUTA = _os.path.join(HERE, "..", "outputs")
OUT = _OUTA if _os.path.isdir(_OUTA) else _os.path.join(HERE, "..", "04outputs")
if not _os.path.isdir(OUT):
    OUT = _os.path.join(HERE, "..", "outputs")


def d1(X):
    return savgol_filter(X, 11, 2, deriv=1, axis=1)


def load_kiwi(target):
    kd = pd.read_parquet(_os.path.join(DATA, "v18_kiwi_instr.parquet")).dropna(
        subset=["SSC", "DM"])
    kw = [c for c in kd.columns if c.startswith("X")]
    ok = ~np.isnan(kd[kw].values.astype(float)).any(0)
    kw = [c for c, b in zip(kw, ok) if b]
    kwl = np.array([float(c[1:]) for c in kw])
    X = d1(kd[kw].values.astype(float)[:, kwl >= 700])
    g = kd["sample_id"].values
    y = kd[target].values.astype(float)
    return X, y, g


def task(a):
    """一个 (target, m, seed) 单元：6 次重复留出，每次重抽噪声并重训。"""
    target, m, seed = a
    X, y_clean, g = load_kiwi(target)
    uq_all = np.unique(g)
    # 果级标签方差 → 按苹果比值拆出 σ_a²、σ_f²（附录 G.1）
    y_fruit = np.array([y_clean[g == f][0] for f in uq_all], float)
    sig_y2 = float(y_fruit.var(ddof=1))
    sig_a2 = sig_y2 / (1.0 + RATIO)
    sig_f2 = RATIO * sig_a2

    rng = np.random.default_rng(seed)
    r2_noisy, r2_clean_eval, ncomp = [], [], []
    for _ in range(NREP):
        # 噪声：按果抽一次，同果所有谱共享（参考测定是逐果做的）
        if m == 0:
            y = y_clean.copy()
            inj = 0.0
        else:
            inj = sig_f2 / m
            eps = rng.normal(0.0, np.sqrt(inj), size=len(uq_all))
            emap = dict(zip(uq_all, eps))
            y = y_clean + np.array([emap[v] for v in g])

        te = set(rng.permutation(uq_all)[: int(round(TEST_FRAC * len(uq_all)))])
        msk = np.array([v in te for v in g])
        Xtr, ytr, gtr = X[~msk], y[~msk], g[~msk]
        # 成分数用**带噪**标签选 —— 真实实验者没有干净标签
        inner = list(GroupKFold(n_splits=INNER).split(Xtr, ytr, gtr))
        best, berr = 2, np.inf
        for c in range(2, min(MAXC, Xtr.shape[1]) + 1, 2):
            e = [np.mean((PLSRegression(c).fit(Xtr[a_], ytr[a_]).predict(Xtr[b_]).ravel()
                          - ytr[b_]) ** 2) for a_, b_ in inner]
            if np.mean(e) < berr - 1e-9:
                berr, best = float(np.mean(e)), c
        p = PLSRegression(best).fit(Xtr, ytr).predict(X[msk]).ravel()
        # 主判据用的是对**带噪标签**的 R²（实验者实际会报告的数）
        r2_noisy.append(float(1 - np.sum((y[msk] - p) ** 2)
                              / np.sum((y[msk] - ytr.mean()) ** 2)))
        # 诊断列：对干净标签的 R²（不参与判据）
        r2_clean_eval.append(float(1 - np.sum((y_clean[msk] - p) ** 2)
                                   / np.sum((y_clean[msk] - y_clean[~msk].mean()) ** 2)))
        ncomp.append(best)
    return {"target": target, "m": m, "seed": seed, "sigma_y2": sig_y2,
            "sigma_a2": sig_a2, "sigma_f2": sig_f2, "inject_var": inj,
            "reps_noisy": r2_noisy, "reps_clean": r2_clean_eval, "ncomp": ncomp}


def boot_mean(by, b=N_BOOT, seed=0):
    ks = [k for k in by if len(by[k])]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        v = []
        for i in rng.choice(len(ks), len(ks), replace=True):
            arr = np.asarray(by[ks[i]], float)
            v.extend(arr[rng.integers(0, len(arr), len(arr))])
        vals.append(np.mean(v))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("A9semisynth_v18 — 预注册 V9（判据见 0NSTATISTICAL_PROTOCOL 附录 G，已锁定）")
    print(f"SEEDS={SEEDS_FORMAL}  m∈{M_LEVELS}(0=∞)  nrep={NREP}  "
          f"σf²/σa²(apple)={RATIO:.6f}")
    print("=" * 78)

    units = [(t, m, s) for t in ("DM", "SSC") for m in M_LEVELS for s in SEEDS_FORMAL]
    with mp.Pool(min(20, max(1, mp.cpu_count() - 2))) as pool:
        res = pool.map(task, units)
    print(f"  {len(units)} 个单元完成 {time.time()-t0:.0f}s", flush=True)

    json.dump({"seeds": SEEDS_FORMAL, "m_levels": M_LEVELS, "nrep": NREP,
               "ratio_apple": RATIO, "runs": res},
              open(_os.path.join(OUT, "A9semisynth_v18_raw.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    rows, verdicts = [], []
    for tgt in ("DM", "SSC"):
        sel = [r for r in res if r["target"] == tgt]
        sig_y2 = sel[0]["sigma_y2"]; sig_a2 = sel[0]["sigma_a2"]; sig_f2 = sel[0]["sigma_f2"]
        clean = [r for r in sel if r["m"] == 0]
        r2_clean = float(np.mean([np.mean(r["reps_noisy"]) for r in clean]))
        devs, p1_fail, p3_hit = [], [], False
        for m in M_LEVELS:
            sub = [r for r in sel if r["m"] == m]
            by = {r["seed"]: r["reps_noisy"] for r in sub}
            obs = float(np.mean([np.mean(v) for v in by.values()]))
            lo, hi = boot_mean(by)
            if m == 0:
                pred, bound, bound_v9 = r2_clean, 1.0, np.nan
            else:
                pred = r2_clean * sig_y2 / (sig_y2 + sig_f2 / m)
                # V9b 修正上界（附录 H.2）：模型可感知整个干净标签，不可感知的只有注入噪声
                bound = sig_y2 / (sig_y2 + sig_f2 / m)
                # V9 原上界（附录 G），并列保留供读者对照——其设定错误已在 H.1 记录
                bound_v9 = SIGMA_A2_APPLE / (SIGMA_A2_APPLE + SIGMA_F2_APPLE / m)
            dev = obs / pred - 1.0 if pred else np.nan
            # P1 判据原文是「每个 m，实测 R² 的 cluster CI 下限不得高于上界」，m=0
            # （干净基线，上界 1.0）本就在「每个 m」之内；早先的实现把整个 m=0 档排除
            # 在 P1 之外，于是「13 档无一越界」里有一档从未被评估器比较过。此处把 P1
            # 拉回判据原文（全 13 档都比较、都落表），**判据本身不变**；devs（P2 跟踪度）
            # 与 P3 绑定性的判据原文限定在有限 m 上，故仍只对有限档累加，口径不变
            # —— 已报汇总数字一个不动。
            # 注意：随本仓库发布的 outputs/A9semisynth_v18.xlsx 产出于本补丁之前，
            # 与重跑结果的差异仅在 m=0 那一行的「V9b 结构上界」列（原写空，现写 1.0）；
            # P1 判定与全部汇总数字不变（干净基线的 CI 下限不可能高于 1.0）。
            if lo > bound:
                p1_fail.append(m)
            if m != 0:
                devs.append(abs(dev))
                if m <= 5 and obs >= P3_FRAC * pred:
                    p3_hit = True
            rows.append({
                "目标": tgt, "m（参考重复次数）": "∞（干净基线）" if m == 0 else m,
                "注入方差 σf²/m": sel[0]["sigma_f2"] / m if m else 0.0,
                "实测 R²（对带噪标签）": obs, "cluster CI95 下限": lo, "cluster CI95 上限": hi,
                "预言 R²_pred": pred, "相对偏差 obs/pred−1": dev,
                "V9b 结构上界（正文口径）": bound,   # m=0 档为 1.0，不再写空
                "V9 原上界（设定错误，仅对照）": bound_v9,
                "P1 是否越界（CI下限>V9b上界）": ("是⚠" if lo > bound else "否"),
                "对照·是否越 V9 原上界": ("是" if (m and lo > bound_v9) else "否"),
                "诊断·对干净标签 R²": float(np.mean([np.mean(r["reps_clean"]) for r in sub])),
                "成分数中位": float(np.median([np.median(r["ncomp"]) for r in sub])),
                "n_seeds": len(by), "n_primary": NREP, "K_inner": INNER})
        med = float(np.median(devs))
        verdicts.append({
            "目标": tgt, "干净基线 R²_clean": r2_clean, "σy²": sig_y2,
            "σa²(标定)": sig_a2, "σf²(标定)": sig_f2,
            "P1 硬证伪·越界的 m": ("无 → 通过" if not p1_fail else f"{p1_fail} ⚠ 未通过"),
            "P2 相对偏差中位": med, "P2 阈值": P2_TOL,
            "P2 判定": "通过" if med <= P2_TOL else "未通过",
            "P3 是否有 m≤5 达 0.8×pred": "是 → 通过" if p3_hit else "否 ⚠ 未通过",
            "总判定": ("通过" if (not p1_fail and med <= P2_TOL and p3_hit) else "未全通过")})

    # ── P4：设计公式的反解可用性（附录 H.3，目标事前锁定）──────────────
    p4rows = []
    idx = {(r["目标"], r["m（参考重复次数）"]): r for r in rows}
    for tgt, rt, blind in P4_TARGETS:
        sel = [r for r in res if r["target"] == tgt]
        sy2, sf2 = sel[0]["sigma_y2"], sel[0]["sigma_f2"]
        r2c = float(np.mean([np.mean(r["reps_noisy"]) for r in sel if r["m"] == 0]))
        m_req_raw = (sf2 / sy2) * rt / (r2c - rt)
        m_req = int(np.ceil(m_req_raw))
        row = idx.get((tgt, m_req))
        if row is None:
            p4rows.append({"目标": tgt, "R²_t": rt, "盲检": blind,
                           "m_req(未取整)": m_req_raw, "m_req": m_req,
                           "实测 R²": np.nan, "判定": "该 m 不在网格内"})
            continue
        obs, lo = row["实测 R²（对带噪标签）"], row["cluster CI95 下限"]
        p4rows.append({
            "目标": tgt, "R²_t（事前指定）": rt, "是否盲检": "盲检" if blind else "已见(不计入)",
            "m_req(未取整)": m_req_raw, "m_req(向上取整)": m_req,
            "该 m 实测 R²": obs, "CI95 下限": lo,
            "缺口 obs−R²_t": obs - rt,
            "P4a 均值口径": "达标" if obs >= rt else "未达标",
            "P4b 严格口径": "达标" if lo >= rt else "未达标"})
    tab_p4 = pd.DataFrame(p4rows)
    blindsel = tab_p4[tab_p4.get("是否盲检", "") == "盲检"]
    n_a = int((blindsel["P4a 均值口径"] == "达标").sum()) if len(blindsel) else 0
    n_b = int((blindsel["P4b 严格口径"] == "达标").sum()) if len(blindsel) else 0
    print("-" * 78)
    for _, r in tab_p4.iterrows():
        print(f"  P4 {r['目标']:3s} 目标R²={r['R²_t（事前指定）']:.2f} [{r['是否盲检']}] "
              f"→ m_req={r['m_req(未取整)']:.2f}→{r['m_req(向上取整)']}  "
              f"实测 {r['该 m 实测 R²']:+.4f} (缺口 {r['缺口 obs−R²_t']:+.4f})  "
              f"均值:{r['P4a 均值口径']} 严格:{r['P4b 严格口径']}")
    print(f"  【P4 汇总·仅盲检 {len(blindsel)} 个目标】均值口径 {n_a}/{len(blindsel)} 达标；"
          f"严格口径 {n_b}/{len(blindsel)} 达标")
    if len(blindsel) and n_a < len(blindsel):
        gaps = blindsel["缺口 obs−R²_t"]
        print(f"  → 按附录 H.3 事前写定的处置：公式给的是**必要非充分**条件，"
              f"实测缺口 {gaps.min():+.4f}~{gaps.max():+.4f}，论文须改述为「m_req 是下限，需加余量」")

    tab_a, tab_b = pd.DataFrame(rows), pd.DataFrame(verdicts)
    for _, r in tab_a.iterrows():
        b = r["V9b 结构上界（正文口径）"]
        print(f"  {r['目标']:3s} m={str(r['m（参考重复次数）']):>8s}  "
              f"实测 {r['实测 R²（对带噪标签）']:+.4f} "
              f"CI[{r['cluster CI95 下限']:+.4f},{r['cluster CI95 上限']:+.4f}]  "
              f"预言 {r['预言 R²_pred']:+.4f}  偏差 {r['相对偏差 obs/pred−1']:+.1%}  "
              f"上界 {'—' if np.isnan(b) else f'{b:.4f}'}  P1越界:{r['P1 是否越界（CI下限>V9b上界）']}")
    print("-" * 78)
    for _, v in tab_b.iterrows():
        print(f"  【{v['目标']}】P1 {v['P1 硬证伪·越界的 m']} | "
              f"P2 中位偏差 {v['P2 相对偏差中位']:.1%} → {v['P2 判定']} | "
              f"P3 {v['P3 是否有 m≤5 达 0.8×pred']} | 总判定：{v['总判定']}")

    sys.path.insert(0, HERE)
    sheets = {0: ("退化轨迹", tab_a), 1: ("预注册 V9b 判定", tab_b),
              2: ("P4 设计公式反解", tab_p4)}
    try:
        from export_utils import write_script_workbook
        out = write_script_workbook(__file__, sheets)
    except Exception as e:
        out = _os.path.join(OUT, "A9semisynth_v18.xlsx")
        print(f"(export_utils 不可用：{e})")
        with pd.ExcelWriter(out) as w:
            for _, (nm, dfr) in sorted(sheets.items()):
                dfr.to_excel(w, sheet_name=nm, index=False)
    print(f"写出 {out}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
