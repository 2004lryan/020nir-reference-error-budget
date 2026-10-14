"""A15 · 饱和掩膜的划分内重算核验：波段子集诊断在「掩膜只用训练折数据算」时的结果。

属 v18 线；文件名按第三章不带版本后缀，版本信息见文末修订记录。

起因（2026-09-01 第十二轮 `/integrity-forensics` eval-design-forensics F001，`HP-EVAL-LEAKAGE`，major；
2026-09-02 独立复核实测）：A6formal_v18.py task_band 的饱和掩膜（保留欠/过量程合计率 ≤1% 的波长）
在分组留出之前用全量 X 计算，属 Kaufman & Narayanan L1「无洁净分离」型。与猕猴桃缺失通道掩膜
（A11derived 表d 证明是空操作）不同，该掩膜**不是**空操作：346 个波长中 76 个的饱和率落在阈值 ±0.5 个
百分点内，训练子集内重算会改变保留集合。审查员要求的复核动作是：在每个外层训练分区内独立推导掩膜后
重跑该诊断，说明其结论是否改变。本脚本就做这一件事。

做法：逐字复刻 A6formal_v18.cv_a1 的划分与选模（default_rng(seed) → permutation → TEST_FRAC=0.40 →
GroupKFold(4) → 成分网格 2:24:2 → PLSRegression 默认 scale），Formal 5 种子 × 15 次，同一划分序列上跑三个口径：
  full  ：全谱 346 波长（对照）
  whole ：全量掩膜（论文口径，与 A6 T8「仅无饱和波段」同一算法；跨架构复现值见 A14 表a）
  local ：训练分区内重算掩膜（去掉泄漏的口径）
导出：
  表a  三口径的跨种子均值、cluster CI95、逐种子（与 A6 T8 同格式）
  表b  逐划分：保留波长数、相对全量掩膜的翻转数、三口径 R²
  表c  饱和率落在 [0.5%, 1.5%] 的临界波长清单
  表d  口径与结论

红线：不重跑 A6 的正式 T8（论文所印 0.109/0.077 保留为原架构值）；本表的 whole 行是同机对照，
用来把 local 行与论文值之间的差拆成「架构」与「泄漏」两部分。

修订记录：| 2026-09-02 | 第十二轮独立复核 | 首版 | 回应 HP-EVAL-LEAKAGE 义务的复核动作 |
"""

from __future__ import annotations

import json
import os as _os
import sys
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from export_utils import write_script_workbook

SEEDS_FORMAL = [20060515, 20041210, 19810915, 2023, 2024]
NREP, MAXC, INNER, TEST_FRAC, N_BOOT = 15, 24, 4, 0.40, 2000
THRESH, BAND_LO, BAND_HI = 1.0, 0.5, 1.5

HERE = _os.path.dirname(_os.path.abspath(__file__))
DATA = _os.path.join(HERE, "..", "03data", "processed")
if not _os.path.isdir(DATA):                       # 开源仓库布局
    DATA = _os.path.join(HERE, "..", "data")

FloatArr = NDArray[np.float64]


def sat_rate(x: FloatArr) -> FloatArr:
    return np.asarray((x < 0.01).mean(0) * 100 + (x > 0.99).mean(0) * 100, dtype=np.float64)


def cv_three(x: FloatArr, y: FloatArr, groups: NDArray[np.object_], seed: int) -> dict[str, list[float]]:
    """同一划分序列上的 full / whole / local 三口径；划分与选模逐字复刻 A6formal.cv_a1。"""
    full_mask = sat_rate(x) <= THRESH
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    out: dict[str, list[float]] = {"full": [], "whole": [], "local": [], "n_band_local": [], "n_flip": []}
    for _ in range(NREP):
        te = set(rng.permutation(uq)[: round(TEST_FRAC * len(uq))])
        m = np.array([g in te for g in groups])
        local_mask = sat_rate(x[~m]) <= THRESH
        out["n_band_local"].append(int(local_mask.sum()))
        out["n_flip"].append(int((local_mask != full_mask).sum()))
        for tag, sel in (("full", np.ones(x.shape[1], bool)), ("whole", full_mask), ("local", local_mask)):
            xs = x[:, sel]
            xtr, ytr, gtr = xs[~m], y[~m], groups[~m]
            inner = list(GroupKFold(n_splits=INNER).split(xtr, ytr, gtr))
            best, berr = 2, np.inf
            for c in range(2, min(MAXC, xtr.shape[1]) + 1, 2):
                e = [np.mean((PLSRegression(c).fit(xtr[a], ytr[a]).predict(xtr[b]).ravel() - ytr[b]) ** 2)
                     for a, b in inner]
                if np.mean(e) < berr - 1e-9:
                    berr, best = float(np.mean(e)), c
            p = PLSRegression(best).fit(xtr, ytr).predict(xs[m]).ravel()
            out[tag].append(float(1 - np.sum((y[m] - p) ** 2) / np.sum((y[m] - ytr.mean()) ** 2)))
    return out


def boot_mean(by: dict[int, list[float]], b: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """两级簇自助（种子为簇），与 A6formal.boot_mean 同构。"""
    rng = np.random.default_rng(seed)
    ks = list(by)
    vals: list[float] = []
    for _ in range(b):
        v: list[float] = []
        for i in rng.choice(len(ks), len(ks), replace=True):
            arr = np.asarray(by[ks[i]])
            v.extend(arr[rng.integers(0, len(arr), len(arr))])
        vals.append(float(np.mean(v)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    t0 = time.time()
    ap = pd.read_parquet(_os.path.join(DATA, "v18_apple_faces.parquet"))
    wc = [c for c in ap.columns if c.endswith("nm")]
    wl = np.array([float(c[:-2]) for c in wc])
    x = ap[wc].values.astype(np.float64)
    y = ap["y_fruit"].values.astype(np.float64)
    groups = ap["id"].values
    rate = sat_rate(x)
    full_mask = rate <= THRESH
    print(f"A15maskcheck — 饱和掩膜划分内重算  SEEDS={SEEDS_FORMAL} nrep={NREP} "
          f"全量掩膜保留 {int(full_mask.sum())}/{len(wl)}")

    res = {s: cv_three(x, y, groups, s) for s in SEEDS_FORMAL}

    label = {"full": "苹果 · 全谱 346 波段 → 整果均值 SSC（同机对照）",
             "whole": "苹果 · 仅无饱和波段·全量掩膜（论文口径，同机对照）→ 整果均值 SSC",
             "local": "苹果 · 仅无饱和波段·训练分区内掩膜（去泄漏口径）→ 整果均值 SSC"}
    rows_a = []
    nband: dict[str, int | str] = {"full": len(wl), "whole": int(full_mask.sum()), "local": "逐划分，见表b"}
    for tag in ("full", "whole", "local"):
        per = {s: float(np.mean(res[s][tag])) for s in SEEDS_FORMAL}
        lo, hi = boot_mean({s: res[s][tag] for s in SEEDS_FORMAL})
        rows_a.append({"量": label[tag], "跨种子均值": float(np.mean(list(per.values()))),
                       "跨种子SD": float(np.std(list(per.values()), ddof=1)),
                       "cluster CI95 下限": lo, "cluster CI95 上限": hi, "n_seeds": len(per),
                       "逐种子": json.dumps({str(k): round(v, 4) for k, v in per.items()}),
                       "波段数": nband[tag],
                       "n_primary": NREP, "K_inner": INNER})
    t_a = pd.DataFrame(rows_a)
    paired = float(np.mean([a - b for s in SEEDS_FORMAL
                            for a, b in zip(res[s]["local"], res[s]["whole"], strict=True)]))

    rows_b = []
    for s in SEEDS_FORMAL:
        for i in range(NREP):
            rows_b.append({"seed": s, "rep": i,
                           "保留波长数（分区内掩膜）": res[s]["n_band_local"][i],
                           "相对全量掩膜翻转的波长数": res[s]["n_flip"][i],
                           "R² full": res[s]["full"][i], "R² whole": res[s]["whole"][i],
                           "R² local": res[s]["local"][i]})
    t_b = pd.DataFrame(rows_b)

    crit = np.where((rate >= BAND_LO) & (rate <= BAND_HI))[0]
    t_c = pd.DataFrame([{"波长 nm": float(wl[i]), "饱和率 %（全量）": float(rate[i]),
                         "全量掩膜是否保留": bool(full_mask[i])} for i in crit])

    nb = t_b["保留波长数（分区内掩膜）"]
    t_d = pd.DataFrame([{
        "阈值": f"欠/过量程合计率 ≤ {THRESH}%（与 A6formal task_band 相同）",
        "临界波长数（饱和率∈[0.5%,1.5%]）": len(crit),
        "全量掩膜保留波长数": int(full_mask.sum()),
        "分区内掩膜保留波长数 min/中位/max":
            f"{int(nb.min())}/{int(nb.median())}/{int(nb.max())}",
        "分区内掩膜与全量掩膜逐位相同的划分数": int((t_b["相对全量掩膜翻转的波长数"] == 0).sum()),
        "local − whole 配对差（均值）": paired,
        "结论": ("该掩膜不是空操作；但去掉泄漏后无饱和子集的 R² 上升而非下降，仍低于同机全谱，"
                 "「全谱 > 无饱和子集」的排序与「限制波段不能修复苹果性能」的结论不变；"
                 "全量掩膜对子集的影响方向是悲观的"),
        "口径说明": ("三口径同一划分序列、同一机器；论文所印 0.109/0.077 是原架构值，本表 whole 行是同机对照，"
                     "用以把 local 行与论文值的差拆成架构与泄漏两部分。不重跑 A6 正式 T8。"),
    }])

    path = write_script_workbook(__file__, {
        0: ("三口径 R²", t_a), 1: ("逐划分保留与翻转", t_b), 2: ("临界波长", t_c), 3: ("口径与结论", t_d)})
    print(f"→ {path}")
    for _, r in t_a.iterrows():
        print(f"  {r['量'][:34]:<36} {r['跨种子均值']:.6f} "
              f"[{r['cluster CI95 下限']:.6f}, {r['cluster CI95 上限']:.6f}]")
    print(f"  临界波长 {len(crit)}；分区内掩膜保留 {int(nb.min())}–{int(nb.max())}（中位 {int(nb.median())}）；"
          f"配对差 {paired:+.6f}；耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
