#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A7datepos_v18 — 「按采集日分组的落差取决于留出的是首尾还是中间日期」这一机制的直接实测。

起因（claim audit R5 · finding C066）：稿件 3.5 声称按采集日分组的 R² 呈双峰，且
「留出中间日期时约 0.7-0.9，留出首尾日期时可低至 -4」。前半句（双峰、低至 -4）可由
A6formal_v18_raw.json 的逐重复值直接读出；**后半句的归因从未被测量**——A6 没有记录
每次留出的是哪几天。把推测写成实测是不可接受的，故本脚本补测。

做法：完全复刻 A6 的 T11 猕猴桃·按采集日分组设置（同评估器、同 5 个 Formal 种子、
同 nrep=6、同随机流），但**额外落盘每次留出的日期集合**，并据此判定该次留出是否
含有「极端日」（day_idx 的最小值或最大值所在的采集日）。随后按「含极端日 / 不含」
分组对照 R²。

这是一个**事前有明确机制预言**的对照：若稿件的归因成立，含极端日的那一组 R² 应显著
更低（外推到训练范围之外的目标档位），不含的那一组应接近按果分组的水平。

[免检] 本脚本不引入新方法，只是给已登记的 M4/M10 补一个记录维度（留出日期集合），
评估器与 A6 的 `cv_a1` 逐行一致。

运行方式:
    python3 code/A7datepos_v18.py

输出文件:
    outputs/A7datepos_v18.xlsx       — 逐次留出明细 + 含/不含极端日的对照
    outputs/A7datepos_v18_raw.json   — 逐次原始值（含留出日期集合）
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEEDS_FORMAL = [20060515, 20041210, 19810915, 2023, 2024]
NREP, MAXC, INNER, TEST_FRAC = 6, 20, 4, 0.40
N_BOOT = 2000

logger = Logger(__file__)


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, ddof=1, keepdims=True) + 1e-12)


def run_seed(X, y, groups, seed, day_of_group):
    """复刻 A6.cv_a1 的随机流，但额外记录每次留出的日期集合。"""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    out = []
    for rep in range(NREP):
        te = set(rng.permutation(uq)[: int(round(TEST_FRAC * len(uq)))])
        m = np.array([g in te for g in groups])
        Xtr, ytr, gtr = X[~m], y[~m], groups[~m]
        inner = list(GroupKFold(n_splits=INNER).split(Xtr, ytr, gtr))
        best, berr = 2, np.inf
        for c in range(2, min(MAXC, Xtr.shape[1]) + 1, 2):
            e = [np.mean((PLSRegression(c).fit(Xtr[a], ytr[a]).predict(Xtr[b]).ravel()
                          - ytr[b]) ** 2) for a, b in inner]
            if np.mean(e) < berr - 1e-9:
                berr, best = float(np.mean(e)), c
        p = PLSRegression(best).fit(Xtr, ytr).predict(X[m]).ravel()
        r2 = float(1 - np.sum((y[m] - p) ** 2) / np.sum((y[m] - ytr.mean()) ** 2))
        te_days = sorted(float(day_of_group[g]) for g in te)
        out.append({"seed": seed, "rep": rep, "R2": r2,
                    "RMSE": float(np.sqrt(np.mean((y[m] - p) ** 2))),
                    "留出日期集合(day_idx)": te_days,
                    "留出天数": len(te_days), "训练天数": len(uq) - len(te_days),
                    "留出目标最小值": min(te_days), "留出目标最大值": max(te_days)})
    return out


def boot_diff(a_pairs, b_pairs, seed=0):
    """两组均值之差的**以种子为簇**的两级自助（先重抽种子，再在种子内重抽次数）。

    2026-07-26 更正（独立一致性审计 HP-SCOPE-INFLATE）：此前按次 i.i.d. 重抽，
    与 §2.3 声明的「一切置信区间均用以种子为簇的两级自助」不一致。同一种子下的多次留出
    共享一条随机流、并非独立样本，i.i.d. 自助会**系统性低估**区间宽度——而这条区间恰恰用于
    支撑「排除 0」，方向上最不保守，故必须按全稿统一口径重算。

    入参为 [(seed, value), ...]，保留簇结构。
    """
    rng = np.random.default_rng(seed)

    def _by_seed(pairs):
        d = {}
        for s_, v in pairs:
            d.setdefault(s_, []).append(float(v))
        return {k: np.asarray(v, float) for k, v in d.items()}

    A, B = _by_seed(a_pairs), _by_seed(b_pairs)
    ka, kb = list(A), list(B)
    out = []
    for _ in range(N_BOOT):
        sa = [A[ka[i]] for i in rng.integers(0, len(ka), len(ka))]
        sb = [B[kb[i]] for i in rng.integers(0, len(kb), len(kb))]
        va = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in sa])
        vb = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in sb])
        out.append(va.mean() - vb.mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    tj = pd.read_parquet("03data/processed/v18_deephs_traj.parquet")
    tj = tj[tj["fruit"] == "Kiwi"].reset_index(drop=True)
    twc = [c for c in tj.columns if c.startswith("w")]
    X = snv(tj[twc].to_numpy(float))
    y = tj["day_idx"].to_numpy(float)
    groups = tj["day"].to_numpy()               # 与 A6 的 T11「Kiwi·按采集日分组」一致

    # 每个采集日对应的 day_idx（目标由分组变量派生 ⇒ 一一对应，此处核验）
    m = tj.groupby("day")["day_idx"].nunique()
    assert (m == 1).all(), "同一采集日出现多个 day_idx，与「目标由分组变量派生」不符"
    day_of_group = tj.groupby("day")["day_idx"].first().to_dict()
    all_days = sorted(day_of_group.values())
    extremes = {min(all_days), max(all_days)}
    logger.log("=" * 74)
    logger.log(f"A7datepos — 猕猴桃 {len(tj)} 条谱 / {len(all_days)} 个采集日；"
               f"day_idx = {all_days}")
    logger.log(f"极端日（day_idx 最小/最大）= {sorted(extremes)}；"
               f"每次留出 {int(round(TEST_FRAC*len(all_days)))} 天")
    logger.log("=" * 74)

    rows = []
    for s in SEEDS_FORMAL:
        rows += run_seed(X, y, groups, s, day_of_group)
    for r in rows:
        r["含极端日"] = bool(extremes & set(r["留出日期集合(day_idx)"]))

    json.dump({"seeds": SEEDS_FORMAL, "nrep": NREP, "K_inner": INNER,
               "all_day_idx": all_days, "extreme_day_idx": sorted(extremes),
               "runs": rows}, open("outputs/A7datepos_v18_raw.json", "w",
                                   encoding="utf-8"), ensure_ascii=False, indent=1)

    det = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v)
                         for k, v in r.items()} for r in rows])

    hi = [r["R2"] for r in rows if not r["含极端日"]]
    lo = [r["R2"] for r in rows if r["含极端日"]]
    hi_p = [(r["seed"], r["R2"]) for r in rows if not r["含极端日"]]
    lo_p = [(r["seed"], r["R2"]) for r in rows if r["含极端日"]]
    d_lo, d_hi = boot_diff(hi_p, lo_p)
    cmp_ = pd.DataFrame([
        {"分组": "留出集**不含**极端日", "n次": len(hi), "R²均值": float(np.mean(hi)),
         "R²最小": float(np.min(hi)), "R²最大": float(np.max(hi)),
         "R²SD": float(np.std(hi, ddof=1))},
        {"分组": "留出集**含**极端日", "n次": len(lo), "R²均值": float(np.mean(lo)),
         "R²最小": float(np.min(lo)), "R²最大": float(np.max(lo)),
         "R²SD": float(np.std(lo, ddof=1))},
        {"分组": "差（不含 − 含）", "n次": len(hi) + len(lo),
         "R²均值": float(np.mean(hi) - np.mean(lo)),
         "R²最小": d_lo, "R²最大": d_hi,
         "R²SD": float("nan")},
    ])
    for _, r in cmp_.iterrows():
        logger.log(f"  {r['分组']:24s} n={int(r['n次']):2d}  R²均值 {r['R²均值']:+.3f}  "
                   f"范围 [{r['R²最小']:+.3f}, {r['R²最大']:+.3f}]")
    logger.log(f"→ 差的 percentile bootstrap CI95 = [{d_lo:+.3f}, {d_hi:+.3f}]；"
               f"{'排除 0，机制归因成立' if d_lo > 0 else '包含 0，机制归因不成立'}")

    out = write_script_workbook(__file__, {
        0: ("逐次留出明细", det),
        1: ("含/不含极端日对照", cmp_),
    })
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
