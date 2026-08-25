#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A5aggregate_v18 — 从 A4formal 的逐重复原始值重新聚合（修正估计量口径）。

起因：A4formal 内对「比值」类量用了**比值之均值** E[X/Y]，而论文原口径与理论预言
对应的是**均值之比** E[X]/E[Y]。二者在本项目下差别很大：分母是 R²≈0.06 的小量，
少数小分母会把 E[X/Y] 炸大（实测泄漏倍数从 1.41 变成 2.28，SD 高达 1.31）。
由 Jensen 不等式，E[X/Y] ≥ E[X]/E[Y]，偏差方向恒为高估。

本脚本只做**聚合**，不重跑任何模型：读 `A4formal_v18_raw.json` 的逐重复原始值，
按两种口径分别给出点估计与 cluster bootstrap CI（簇=种子），并列报告以便对照。
论文正文一律采用**均值之比**（与理论预言、与原稿口径一致）。

[免检] 纯聚合与格式化，不产生任何新的模型拟合。

输出：outputs/A5aggregate_v18.xlsx
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

RAW = "outputs/A4formal_v18_raw.json"
N_BOOT = 2000
V8_RATIO = 1.5

# 93sscceiling 表c（闭式矩估计，与种子无关）
SIGMA_A2, SIGMA_F2 = 2.327025, 2.551483
VAR_FACE = SIGMA_A2 + SIGMA_F2
VAR_FRUIT5 = SIGMA_A2 + SIGMA_F2 / 5

logger = Logger(__file__)


def boot_ratio_of_means(num: dict, den: dict, b=N_BOOT, seed=0):
    """簇=种子的两级自助，统计量 = 均值之比 E[X]/E[Y]（分子分母配对重抽）。"""
    seeds = [s for s in num if len(num[s]) and len(den[s])]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        ns, ds = [], []
        for i in pick:
            s = seeds[i]
            a, c = np.asarray(num[s], float), np.asarray(den[s], float)
            k = min(len(a), len(c))
            idx = rng.integers(0, k, k)          # 分子分母用同一 idx → 保持配对
            ns.extend(a[:k][idx]); ds.extend(c[:k][idx])
        if ds and np.mean(ds) != 0:
            vals.append(np.mean(ns) / np.mean(ds))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def boot_mean(by: dict, b=N_BOOT, seed=0):
    seeds = [s for s in by if len(by[s])]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        v = []
        for i in pick:
            a = np.asarray(by[seeds[i]], float)
            v.extend(a[rng.integers(0, len(a), len(a))])
        if v:
            vals.append(np.mean(v))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def ratio_row(label, num, den, extra=None):
    per_seed = {s: np.mean(num[s]) / np.mean(den[s]) for s in num
                if len(num[s]) and len(den[s]) and np.mean(den[s]) != 0}
    # 点估计取**合并**均值之比，与 boot_ratio_of_means 的自助统计量严格同型；
    # 若改用「逐种子比值再平均」，点估计与 CI 会来自两个不同的估计量，读者无法核对。
    rom = float(np.mean(np.concatenate([np.asarray(num[s], float) for s in per_seed]))
                / np.mean(np.concatenate([np.asarray(den[s], float) for s in per_seed])))
    lo, hi = boot_ratio_of_means(num, den)
    # 对照：比值之均值（说明其不稳定，不用于正文）
    moc = {s: [a / c for a, c in zip(num[s], den[s]) if c > 0] for s in num}
    moc_mean = float(np.mean([np.mean(v) for v in moc.values() if len(v)]))
    d = {"量": label,
         "均值之比（正文口径）": rom,
         "cluster CI95 下限": lo, "cluster CI95 上限": hi,
         "逐种子比值SD（离散度）": float(np.std(list(per_seed.values()), ddof=1)),
         "对照·比值之均值（不稳，不入正文）": moc_mean,
         "逐种子": json.dumps({str(k): round(v, 4) for k, v in per_seed.items()}, ensure_ascii=False)}
    if extra:
        d.update(extra)
    return d


def main() -> None:
    raw = json.load(open(RAW, encoding="utf-8"))
    seeds = raw["seeds"]
    logger.log("=" * 74)
    logger.log(f"A5aggregate_v18 — 重新聚合（种子 {seeds}，n_primary={raw['n_primary']}，"
               f"K_inner={raw['K_inner']}）")
    logger.log("正文口径 = 均值之比 E[X]/E[Y]；比值之均值仅作对照")
    logger.log("=" * 74)

    apple = raw["apple"]
    leak = raw["leak"]
    deep = raw["deephs"]

    def pick(unit_list, **kw):
        return [r for r in unit_list if all(r.get(k) == v for k, v in kw.items())]

    # ── 表a：苹果光谱 R²（各预处理 × 各目标）─────────────────────────
    rows_a = []
    for pre in ("raw", "SNV", "1stDeriv", "SNV+1stDeriv"):
        for tgt, nm in (("y_face", "面级 SSC"), ("y_fruit", "整果均值 SSC"), ("y_dev", "面级偏离")):
            by = {r["seed"]: r["reps"][tgt] for r in pick(apple, preproc=pre)}
            per = {s: float(np.mean(v)) for s, v in by.items()}
            lo, hi = boot_mean(by)
            rows_a.append({"预处理": pre, "目标": nm,
                           "跨种子均值 R²": float(np.mean(list(per.values()))),
                           "跨种子SD": float(np.std(list(per.values()), ddof=1)),
                           "cluster CI95 下限": lo, "cluster CI95 上限": hi,
                           "n_seeds": len(per), "n_primary": raw["n_primary"],
                           "K_inner": raw["K_inner"]})
    tab_a = pd.DataFrame(rows_a)

    # ── 表b：衰减比（均值之比）───────────────────────────────────────
    nf = {r["seed"]: r["reps"]["y_face"] for r in pick(apple, preproc="raw")}
    nu = {r["seed"]: r["reps"]["y_fruit"] for r in pick(apple, preproc="raw")}
    r_att = ratio_row("衰减比 = R²(面级) / R²(5面均值)，raw", nf, nu)
    tab_b = pd.DataFrame([
        r_att,
        {"量": "理论值B：(σ²a+σ²f/5)/(σ²a+σ²f) ← 本文 5 面均值目标的正确预言",
         "均值之比（正文口径）": VAR_FRUIT5 / VAR_FACE,
         "逐种子": "闭式，与种子无关"},
        {"量": "理论值A：ICC = σ²a/(σ²a+σ²f) ← 仅整果目标无噪(m→∞)时成立",
         "均值之比（正文口径）": SIGMA_A2 / VAR_FACE, "逐种子": "闭式，与种子无关"},
    ])
    inB = r_att["cluster CI95 下限"] <= VAR_FRUIT5 / VAR_FACE <= r_att["cluster CI95 上限"]
    inA = r_att["cluster CI95 下限"] <= SIGMA_A2 / VAR_FACE <= r_att["cluster CI95 上限"]
    logger.log(f"[衰减比] 实测 {r_att['均值之比（正文口径）']:.4f} "
               f"CI95 [{r_att['cluster CI95 下限']:.4f}, {r_att['cluster CI95 上限']:.4f}]")
    logger.log(f"         理论B={VAR_FRUIT5/VAR_FACE:.4f} 在CI内? {inB}  |  "
               f"理论A(ICC)={SIGMA_A2/VAR_FACE:.4f} 在CI内? {inA}")

    # ── 表c：泄漏倍数（均值之比）────────────────────────────────────
    rows_c = []
    for pre in ("raw", "1stDeriv"):
        for tgt in ("y_face", "y_fruit"):
            g = {r["seed"]: r["reps"][f"grouped|{tgt}"] for r in pick(leak, preproc=pre)}
            rd = {r["seed"]: r["reps"][f"random|{tgt}"] for r in pick(leak, preproc=pre)}
            rows_c.append(ratio_row(f"{pre} · {tgt} · 虚高倍数（按行随机 / 按果分组）", rd, g,
                                    {"预处理": pre, "目标": tgt}))
    tab_c = pd.DataFrame(rows_c)
    lo_r = tab_c["均值之比（正文口径）"].min(); hi_r = tab_c["均值之比（正文口径）"].max()
    logger.log(f"[泄漏] 四种组合虚高 {lo_r:.2f}–{hi_r:.2f} 倍"
               f"（原稿单种子口径为 1.45–1.69）")

    # ── 表d：V8（均值之比）──────────────────────────────────────────
    rows_d = []
    for tgt in ("firmness", "storage_days"):
        for fr in ("Kiwi", "Avocado", "__汇总__"):
            sel = pick(deep, target=tgt, fruit=fr)
            if not sel:
                continue
            gf = {r["seed"]: r["reps"]["按果分组"] for r in sel}
            gd = {r["seed"]: r["reps"]["按采集日分组"] for r in sel}
            row = ratio_row(f"{tgt} · {fr} · RMSE(按天)/RMSE(按果)", gd, gf,
                            {"目标": tgt, "物种": fr})
            m, lo = row["均值之比（正文口径）"], row["cluster CI95 下限"]
            row["V8 阈值"] = V8_RATIO
            row["判定·均值口径"] = "达标" if m >= V8_RATIO else "未达标"
            row["判定·CI下限口径(更严)"] = "达标" if lo >= V8_RATIO else "未达标"
            rows_d.append(row)
            logger.log(f"[V8] {tgt:13s} {fr:9s} 比值={m:.3f} CI95[{lo:.3f}, "
                       f"{row['cluster CI95 上限']:.3f}] → {row['判定·均值口径']} / "
                       f"{row['判定·CI下限口径(更严)']}(严)")
    tab_d = pd.DataFrame(rows_d)

    n_pass = int((tab_d["判定·均值口径"] == "达标").sum())
    n_pass_strict = int((tab_d["判定·CI下限口径(更严)"] == "达标").sum())
    logger.log("=" * 74)
    logger.log(f"V8 汇总：均值口径 {n_pass}/{len(tab_d)} 达标；严格 CI 口径 "
               f"{n_pass_strict}/{len(tab_d)} 达标 → 撤回日级泄漏主张的决定在 Formal 5 种子下"
               f"{'依然成立' if n_pass_strict <= 2 else '需重新审视'}")

    out = write_script_workbook(__file__, {
        0: ("苹果光谱 R²（5 种子）", tab_a),
        1: ("衰减比（均值之比）", tab_b),
        2: ("泄漏倍数（均值之比）", tab_c),
        3: ("V8 判定（均值之比）", tab_d),
    })
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
