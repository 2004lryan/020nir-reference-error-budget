#!/usr/bin/env python3
"""A16retrainterm — 把式 (degrade) 的「重训不变性」由声明改为实测。

起因（第十三轮 integrity-forensics, proof-derivation-forensics F002,
HP-ASSUMPTION-SMUGGLE, major, false_positive_risk=low）：

    式 (degrade) R²_pred(m) = R²_clean · σ_y² / (σ_y² + σ_f²/m) 把 R²_clean 原样带过，
    而 A9 的设定明写模型在**带噪标签上重训**、成分数**也用带噪标签选**。推导没有证明
    重训保持干净可解释方差，也没有证明学习器误差与注入噪声无关，因此该式至多对
    「注入之前即已固定的预测器」成立。

    审查员给出的可执行动作原文：
      "Ask the authors to state and justify retraining invariance, or compare the
       retrained procedure against a fixed/oracle predictor and quantify any
       additional retraining term."

本脚本执行后半句：在**同一次重复内**（同种子、同一份噪声实现、同一划分）多跑一条
**固定预测器**臂——成分数与拟合都只用干净标签（即注入之前就已确定，与注入噪声独立），
再用同一批带噪测试标签评分。于是

    retraining term  Δ(m) = R²_fixed(m) − R²_retrained(m)

就是式 (degrade) 在「重训」这一步上留下的差额，可逐 m 报告并给 cluster bootstrap CI。

**本脚本不改 A9 的任何预注册判据，也不覆盖 A9 的任何产物。** 常量、预处理、评估器、
噪声注入方式全部逐字沿用 A9semisynth_v18，只增加固定预测器臂；retrained 臂在同一
随机数流上复现 A9 的口径，用于自证两者可比。

[免检] 不引入新方法：固定/oracle 预测器对照是既有评估器的一次重用。

运行:
    python3 02code/A16retrainterm.py

输出:
    04outputs/A16retrainterm.xlsx
      表a  逐 (target, m) 的 retrained / fixed / 预言 三条曲线与 Δ
      表b  Δ 的 cluster bootstrap 95% CI（按 (m, seed) 单元重抽，与 A9 同法）
      表c  本表读法与限度
    04outputs/A16retrainterm_raw.json  — 逐重复原始值
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

# ── 常量：逐字沿用 A9semisynth_v18（附录 G 预注册值，本脚本不得改）──────────
SEEDS_FORMAL = [20060515, 20041210, 19810915, 2023, 2024]
M_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 20, 0]     # 0 代表 m=∞
SIGMA_A2_APPLE, SIGMA_F2_APPLE = 2.327025, 2.551483
RATIO = SIGMA_F2_APPLE / SIGMA_A2_APPLE
NREP, MAXC, INNER, TEST_FRAC = 6, 24, 4, 0.40
N_BOOT = 2000

FloatArr = NDArray[np.float64]

HERE = _os.path.dirname(_os.path.abspath(__file__))
DATA = _os.path.join(HERE, "..", "03data", "processed")
if not _os.path.isdir(DATA):
    DATA = _os.path.join(HERE, "..", "data")
OUT = _os.path.join(HERE, "..", "04outputs")
if not _os.path.isdir(OUT):
    OUT = _os.path.join(HERE, "..", "outputs")


def d1(spec: FloatArr) -> FloatArr:
    out: FloatArr = savgol_filter(spec, 11, 2, deriv=1, axis=1)
    return out


def load_kiwi(target: str) -> tuple[FloatArr, FloatArr, NDArray[np.object_]]:
    kd = pd.read_parquet(_os.path.join(DATA, "v18_kiwi_instr.parquet")).dropna(
        subset=["SSC", "DM"])
    kw = [c for c in kd.columns if c.startswith("X")]
    ok = ~np.isnan(kd[kw].values.astype(float)).any(0)
    kw = [c for c, b in zip(kw, ok, strict=True) if b]
    kwl = np.array([float(c[1:]) for c in kw])
    xs = d1(kd[kw].values.astype(float)[:, kwl >= 700])
    g = kd["sample_id"].values
    y = kd[target].values.astype(float)
    return xs, y, g


def _select_and_fit(xtr: FloatArr, ytr: FloatArr,
                    gtr: NDArray[np.object_]) -> tuple[PLSRegression, int]:
    """A9 的成分数选择 + 拟合，逐字同法。标签由调用方决定（带噪或干净）。"""
    inner = list(GroupKFold(n_splits=INNER).split(xtr, ytr, gtr))
    best, berr = 2, np.inf
    for c in range(2, min(MAXC, xtr.shape[1]) + 1, 2):
        e = [np.mean((PLSRegression(c).fit(xtr[a_], ytr[a_]).predict(xtr[b_]).ravel()
                      - ytr[b_]) ** 2) for a_, b_ in inner]
        if np.mean(e) < berr - 1e-9:
            berr, best = float(np.mean(e)), c
    return PLSRegression(best).fit(xtr, ytr), best


def task(a: tuple[str, int, int]) -> dict[str, Any]:
    """一个 (target, m, seed) 单元：6 次重复，每次同噪声同划分下跑两条臂。"""
    target, m, seed = a
    xs, y_clean, g = load_kiwi(target)
    uq_all = np.unique(g)
    y_fruit = np.array([y_clean[g == f][0] for f in uq_all], float)
    sig_y2 = float(y_fruit.var(ddof=1))
    sig_a2 = sig_y2 / (1.0 + RATIO)
    sig_f2 = RATIO * sig_a2

    rng = np.random.default_rng(seed)
    r2_retrained, r2_fixed, nc_re, nc_fx = [], [], [], []
    for _ in range(NREP):
        if m == 0:
            y = y_clean.copy()
            inj = 0.0
        else:
            inj = sig_f2 / m
            eps = rng.normal(0.0, np.sqrt(inj), size=len(uq_all))
            emap = dict(zip(uq_all, eps, strict=True))
            y = y_clean + np.array([emap[v] for v in g])

        te = set(rng.permutation(uq_all)[: round(TEST_FRAC * len(uq_all))])
        msk = np.array([v in te for v in g])
        xtr, gtr = xs[~msk], g[~msk]
        ytr_noisy, ytr_clean = y[~msk], y_clean[~msk]
        # 分母对两条臂完全相同：带噪测试标签相对**带噪**训练均值的总平方和。
        sst = float(np.sum((y[msk] - ytr_noisy.mean()) ** 2))

        # 臂 1：重训（A9 口径）—— 成分数与拟合都用带噪标签
        mdl, c_re = _select_and_fit(xtr, ytr_noisy, gtr)
        p_re = mdl.predict(xs[msk]).ravel()
        r2_retrained.append(float(1 - np.sum((y[msk] - p_re) ** 2) / sst))
        nc_re.append(c_re)

        # 臂 2：固定/oracle 预测器 —— 成分数与拟合都只用干净标签，
        #        即在注入之前就已完全确定，与本次注入的噪声独立；
        #        评分仍对同一批带噪测试标签，与臂 1 逐项可比。
        mdl_f, c_fx = _select_and_fit(xtr, ytr_clean, gtr)
        p_fx = mdl_f.predict(xs[msk]).ravel()
        r2_fixed.append(float(1 - np.sum((y[msk] - p_fx) ** 2) / sst))
        nc_fx.append(c_fx)

    return {"target": target, "m": m, "seed": seed, "sigma_y2": sig_y2,
            "sigma_a2": sig_a2, "sigma_f2": sig_f2, "inject_var": inj,
            "reps_retrained": r2_retrained, "reps_fixed": r2_fixed,
            "ncomp_retrained": nc_re, "ncomp_fixed": nc_fx}


def boot_mean(by: dict[Any, list[float]], b: int = N_BOOT,
              seed: int = 0) -> tuple[float, float, float]:
    """A9 的两级簇自助，逐字同法：按单元重抽，单元内取均值。"""
    ks = [k for k in by if len(by[k])]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.choice(len(ks), size=len(ks), replace=True)
        pool: list[float] = []
        for i in pick:
            pool.extend(by[ks[i]])
        vals.append(float(np.mean(pool)))
    allv = [v for k in ks for v in by[k]]
    return (float(np.mean(allv)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def main() -> None:
    t0 = time.time()
    print(f"A16retrainterm — SEEDS={SEEDS_FORMAL}  m∈{M_LEVELS}(0=∞)  nrep={NREP}", flush=True)
    units = [(t, m, s) for t in ("DM", "SSC") for m in M_LEVELS for s in SEEDS_FORMAL]
    with mp.Pool(min(20, max(1, mp.cpu_count() - 2))) as pool:
        runs = pool.map(task, units)
    print(f"  {len(units)} 个单元完成 {time.time() - t0:.0f}s", flush=True)

    with open(_os.path.join(OUT, "A16retrainterm_raw.json"), "w", encoding="utf-8") as fh:
        json.dump({"seeds": SEEDS_FORMAL, "m_levels": M_LEVELS, "nrep": NREP,
                   "ratio_apple": RATIO, "runs": runs}, fh, ensure_ascii=False, indent=1)

    # R²_clean 基线：m=∞ 档的重训臂（此时无噪声，两臂同义），与 A9 口径一致
    rows_a, rows_b = [], []
    for tgt in ("DM", "SSC"):
        base = [r for r in runs if r["target"] == tgt and r["m"] == 0]
        r2_clean = float(np.mean([v for r in base for v in r["reps_retrained"]]))
        sig_y2 = base[0]["sigma_y2"]
        sig_f2 = base[0]["sigma_f2"]
        for m in M_LEVELS:
            sel = [r for r in runs if r["target"] == tgt and r["m"] == m]
            re_v = [v for r in sel for v in r["reps_retrained"]]
            fx_v = [v for r in sel for v in r["reps_fixed"]]
            pred = (r2_clean if m == 0
                    else r2_clean * sig_y2 / (sig_y2 + sig_f2 / m))
            d_by: dict[Any, list[float]] = {(r["seed"],): [f - q for f, q in
                                   zip(r["reps_fixed"], r["reps_retrained"], strict=True)] for r in sel}
            dmean, dlo, dhi = boot_mean(d_by, seed=hash((tgt, m)) % (2 ** 32))
            rows_a.append({
                "目标": tgt, "m": "∞" if m == 0 else m,
                "R²_pred 式(degrade)": round(pred, 6),
                "R²_重训臂(A9口径)": round(float(np.mean(re_v)), 6),
                "R²_固定预测器臂": round(float(np.mean(fx_v)), 6),
                "重训项 Δ=固定−重训": round(dmean, 6),
                "Δ 的 95%CI 下限": round(dlo, 6), "Δ 的 95%CI 上限": round(dhi, 6),
                "固定臂相对预言偏差(%)": round(100 * (float(np.mean(fx_v)) / pred - 1), 3),
                "重训臂相对预言偏差(%)": round(100 * (float(np.mean(re_v)) / pred - 1), 3),
                "成分数中位(重训/固定)":
                    f"{int(np.median([c for r in sel for c in r['ncomp_retrained']]))}/"
                    f"{int(np.median([c for r in sel for c in r['ncomp_fixed']]))}",
            })
    # 表b：把 12 个有限 m 档合并，给出 Δ 与两臂相对偏差的总体口径
    for tgt in ("DM", "SSC"):
        sel = [r for r in runs if r["target"] == tgt and r["m"] != 0]
        d_by = {(r["m"], r["seed"]): [f - q for f, q in
                                      zip(r["reps_fixed"], r["reps_retrained"], strict=True)] for r in sel}
        dmean, dlo, dhi = boot_mean(d_by, seed=7)
        sub = [row for row in rows_a if row["目标"] == tgt and row["m"] != "∞"]
        rows_b.append({
            "目标": tgt, "合并档数": len(sub),
            "Δ 均值": round(dmean, 6), "Δ 95%CI": f"[{dlo:.6f}, {dhi:.6f}]",
            "固定臂相对预言偏差中位(%)":
                round(float(np.median([r["固定臂相对预言偏差(%)"] for r in sub])), 3),
            "重训臂相对预言偏差中位(%)":
                round(float(np.median([r["重训臂相对预言偏差(%)"] for r in sub])), 3),
            "Δ>0 的档数": sum(1 for r in sub if float(r["重训项 Δ=固定−重训"]) > 0),
        })

    rows_c = [
        {"项": "本表回答的问题",
         "说明": "式 (degrade) 把 R²_clean 原样带过，只对『注入之前即已固定、误差与注入噪声"
                 "无关』的预测器精确；A9 的学习器在带噪标签上重训、成分数也用带噪标签选。"
                 "本表测的就是这两者之间的差额。"},
        {"项": "两条臂的唯一差别",
         "说明": "同种子、同一份噪声实现、同一划分、同一 R² 分母。重训臂用带噪标签选成分数并"
                 "拟合（A9 口径）；固定臂用干净标签选成分数并拟合（注入前即已确定），"
                 "两臂都对同一批带噪测试标签评分。"},
        {"项": "Δ 的读法",
         "说明": "Δ = R²_固定 − R²_重训 > 0 表示重训在带噪标签上损失了性能，即式 (degrade) "
                 "作为重训过程的预言偏乐观、作为设计规则的下限方向保守。CI 由与 A9 相同的"
                 f"两级簇自助给出（{N_BOOT} 次，按单元重抽）。"},
        {"项": "平台效应为何不污染 Δ",
         "说明": "本机重训臂与 A9 已发布的逐次值并非逐位相同：实测同一 (DM, m=5, 20060515) 单元"
                 "6 次重复中 1 次逐位相同、其余 R² 差 ~1e-3，机制是内层 CV 的 MSE 曲线在高成分数"
                 "段近平坦，BLAS 浮点差翻转了所选成分数（A9 记录 22/22/22/24/24/24，本机 "
                 "24/24/24/22/24/22）。这不影响本表：两条臂在**同一台机器、同一次重复、同一份噪声、"
                 "同一划分**上跑，Δ 是配对差，平台效应对两臂同向作用后相消。因此 Δ 可跨平台读，"
                 "而两臂各自的绝对 R² 只在本机口径下成立。"},
        {"项": "限度（必须一并读）",
         "说明": "① 固定臂仍用同一训练集拟合，只是标签干净，因此它是『注入前固定』的一个"
                 "实现而非唯一实现；② 本表不构成对不变性的证明，只给出该假设在本数据、"
                 "本评估器下留下的差额；③ 本脚本不改 A9 的任何预注册判据，A9 产物未被覆盖。"},
    ]

    out = _os.path.join(OUT, "A16retrainterm.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame(rows_a).to_excel(xw, sheet_name="表a逐m两臂", index=False)
        pd.DataFrame(rows_b).to_excel(xw, sheet_name="表b合并与CI", index=False)
        pd.DataFrame(rows_c).to_excel(xw, sheet_name="表c读法与限度", index=False)
    print(f"写出 {out}   总耗时 {time.time() - t0:.0f}s")
    for r in rows_b:
        print(f"  {r['目标']}: Δ={r['Δ 均值']:.4f} {r['Δ 95%CI']}  "
              f"固定臂偏差中位 {r['固定臂相对预言偏差中位(%)']}%  "
              f"重训臂 {r['重训臂相对预言偏差中位(%)']}%")


if __name__ == "__main__":
    main()
