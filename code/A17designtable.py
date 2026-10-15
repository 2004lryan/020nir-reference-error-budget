#!/usr/bin/env python3
"""A17designtable — 设计表（3／5／10／21）与单面取值 SD（2.209）落盘。

起因：稿件 §2.5 写明，正文报告、由其他已存储分量派生的量也各自落盘，「可追溯」指已存储，
不只是可重算。§3.5 的设计表（R²max = 0.7／0.8／0.9／0.95 → m ≥ 3／5／10／21）与 §3.1 的
单面取值 SD（2.209 °Brix）都由 93sscceiling_v18.xlsx 表c 的方差分量现算。此前设计表只作为
稿件值的重放核对行出现在 A14reprocheck_f1_v18.xlsx 表a（只有上取整后的 m），两分量合成的单面
取值 SD 没有落盘：A14 表a 那一行是配上光谱的 3,149 个面的样本 SD 2.207，记于稿件改用两分量
口径之前（该表表b 有说明）；A11derived_v18 表a 存了两个方差分量之比。

本脚本只读已落盘的分量，按正文的式子算出这两组量并写入自己的工作簿：不重算方差分量，
不拟合任何模型。方差分量由 93sscceiling_v18.py 从苹果糖度原始表（633 果 × 5 面）算出；
另用 03data/processed/v18_apple_faces.parquet（配上光谱的 3,149 个面）核对它能确定的部分。

[免检] 不产生新结论：设计表与单面 SD 都是已存储分量的闭式函数。

输入:
    04outputs/93sscceiling_v18.xlsx  表c 行「全体（主口径）」；表e 行「全体」、m_faces_as_truth=1
    03data/processed/v18_apple_faces.parquet  （核对用；没有这份文件时表d 只写一行说明）

输出:
    04outputs/A17designtable.xlsx
      表a  方差分量与比值      表b  设计表          表c  单面取值 SD 的三种口径
      表d  与 parquet 的核对   表e  读法与限度
    正文所印值有任何一项对不上时以返回码 1 退出。

运行:
    python3 02code/A17designtable.py
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import write_script_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "04outputs")
if not os.path.isdir(OUT):
    OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "03data", "processed")
if not os.path.isdir(DATA):
    DATA = os.path.join(ROOT, "data")
WB93 = os.path.join(OUT, "93sscceiling_v18.xlsx")
SHEET_VC = "表c：方差分解（总体与分产地）"
SHEET_CEIL = "表e：可达上界 R²_RMSE"
APPLE = os.path.join(DATA, "v18_apple_faces.parquet")

K = 5                                                     # 每果 5 个面
TARGETS = ((0.70, 3), (0.80, 5), (0.90, 10), (0.95, 21))  # (R²max, 正文所印 m)
TEXT_RATIO = 1.0965                                       # 正文所印 σf²/σa²
TEXT_SD = 2.209                                           # 正文所印单面取值 SD

FloatArr = NDArray[np.float64]


def m_required(ratio: float, r2: float) -> tuple[float, int]:
    """式 (eq:design)：m ≥ (σf²/σa²)·R²max/(1−R²max)；返回未取整值与上取整。"""
    raw = ratio * r2 / (1.0 - r2)
    return raw, math.ceil(raw - 1e-9)


def ceiling(ratio: float, m: int) -> float:
    """真值取 m 个面均值时的可达上界 σa²/(σa²+σf²/m) = 1/(1+比值/m)。"""
    return 1.0 / (1.0 + ratio / m)


def anova(mat: FloatArr) -> dict[str, float]:
    """平衡单向随机效应 ANOVA 矩估计，与 93sscceiling_v18.variance_components 同式。"""
    n, k = mat.shape
    mu = mat.mean(axis=1)
    msb = float(k * ((mu - mat.mean()) ** 2).sum() / (n - 1))
    msw = float(((mat - mu[:, None]) ** 2).sum() / (n * (k - 1)))
    return {"n": float(n), "MSB": msb, "MSW": msw, "var_apple": max((msb - msw) / k, 0.0),
            "var_face": msw}


def table_ab(vc: pd.Series) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    msb, msw = float(vc["MS_between"]), float(vc["MS_within"])
    va, vf = float(vc["var_apple"]), float(vc["var_face"])
    ratio = vf / va
    src = f"93sscceiling_v18.xlsx {SHEET_VC} · 行「全体（主口径）」"
    rows_a = [
        {"量": "果数 n", "全精度值": int(vc["n_fruit"]), "正文所印值": 633,
         "是否一致": int(vc["n_fruit"]) == 633, "来源": src},
        {"量": "果间均方 MS_between", "全精度值": msb, "正文所印值": "—", "是否一致": "—", "来源": src},
        {"量": "果内均方 MS_within", "全精度值": msw, "正文所印值": "—", "是否一致": "—", "来源": src},
        {"量": "σ̂a²（果间）", "全精度值": va, "正文所印值": 2.327, "是否一致": round(va, 3) == 2.327,
         "来源": src + "；等于 (MS_between − MS_within)/5，差 "
                       f"{abs(va - (msb - msw) / K):.1e}"},
        {"量": "σ̂f²（果内）", "全精度值": vf, "正文所印值": 2.551, "是否一致": round(vf, 3) == 2.551,
         "来源": src + "；等于 MS_within"},
        {"量": "σ̂f²/σ̂a²", "全精度值": ratio, "正文所印值": TEXT_RATIO,
         "是否一致": round(ratio, 4) == TEXT_RATIO,
         "来源": "由上两行的全精度值相除；A11derived_v18.xlsx 表a 存有同一比值"},
    ]
    rows_b = []
    for r2, m_text in TARGETS:
        raw, m = m_required(ratio, r2)
        odds = (1.0 - r2) / r2
        rows_b.append({
            "R²max": r2, "m 未取整 比值·R²/(1−R²)": raw, "m 上取整": m, "正文所印 m": m_text,
            "是否一致": m == m_text,
            "m 处的上界 1/(1+比值/m)": ceiling(ratio, m),
            "m−1 处的上界": ceiling(ratio, m - 1) if m > 1 else float("nan"),
            "m 不变的比值区间 下端（开）": (m - 1) * odds,
            "m 不变的比值区间 上端（闭）": m * odds,
        })
    return rows_a, rows_b, ratio


def table_c(vc: pd.Series, var_truth_m1: float, ssc_paired: FloatArr | None) -> list[dict[str, Any]]:
    msb, msw = float(vc["MS_between"]), float(vc["MS_within"])
    va, vf, n = float(vc["var_apple"]), float(vc["var_face"]), int(vc["n_fruit"])
    sd_model = math.sqrt(va + vf)
    sd_truth = math.sqrt(var_truth_m1)
    tss = (n - 1) * msb + n * (K - 1) * msw
    sd_cells = math.sqrt(tss / (n * K - 1))
    rows: list[dict[str, Any]] = [
        {"口径": "由两分量合成：√(σ̂a² + σ̂f²)（正文所用）", "全精度值": sd_model,
         "三位小数": round(sd_model, 3), "正文所印值": TEXT_SD, "是否一致": round(sd_model, 3) == TEXT_SD,
         "算法与来源": f"93 {SHEET_VC} 行「全体（主口径）」var_apple + var_face 开方"},
        {"口径": "同上，由 93 表e 单面真值方差开方（恒等核对）", "全精度值": sd_truth,
         "三位小数": round(sd_truth, 3), "正文所印值": TEXT_SD, "是否一致": round(sd_truth, 3) == TEXT_SD,
         "算法与来源": f"93 {SHEET_CEIL} 行「全体」m=1 的 var_of_truth 开方；"
                       f"与上一行差 {abs(sd_truth - sd_model):.1e}"},
        {"口径": f"{n * K:,} 个面值的样本 SD（ddof=1）", "全精度值": sd_cells,
         "三位小数": round(sd_cells, 3), "正文所印值": "—", "是否一致": "—",
         "算法与来源": "平衡单向设计的总平方和 = (n−1)·MS_between + n(k−1)·MS_within，"
                       f"除以 nk−1 = {n * K - 1} 再开方；两均方取自 93 表c"},
    ]
    if ssc_paired is not None:
        sd_pair = float(np.std(ssc_paired, ddof=1))
        rows.append({
            "口径": f"配上光谱的 {len(ssc_paired):,} 个面值的样本 SD（ddof=1）", "全精度值": sd_pair,
            "三位小数": round(sd_pair, 3), "正文所印值": "—", "是否一致": "—",
            "算法与来源": "v18_apple_faces.parquet 的 ssc 列；A14reprocheck_f1_v18.xlsx 表a "
                          "「单面取值 SD（光谱配对 3149 行口径）」一行按此口径核对"})
    return rows


def parquet_checks(vc: pd.Series, ratio: float) -> tuple[list[dict[str, Any]], FloatArr]:
    """parquet 能确定的部分逐项核对 93 表c；返回表d 各行与配对面值的 ssc。

    parquet 只有配上光谱的面。缺面的果，其 y_fruit 是已配面值的均值（表d 有实测），未配上光谱的
    面值在 parquet 里没有，所以 93 的两个均方不能由它逐位复算，只能核对必要条件与五面齐全的子集。
    """
    ap = pd.read_parquet(APPLE, columns=["id", "face", "ssc", "y_fruit"])
    n93, msb93, msw93 = int(vc["n_fruit"]), float(vc["MS_between"]), float(vc["MS_within"])
    va93, vf93 = float(vc["var_apple"]), float(vc["var_face"])
    rows: list[dict[str, Any]] = []

    def add(item: str, value: Any, ref: Any, ok: Any, note: str = "") -> None:
        if isinstance(ok, np.bool_):
            ok = bool(ok)
        rows.append({"核对项": item, "parquet 实测": value, "对照值": ref, "是否一致": ok, "说明": note})

    grp = ap.groupby("id", sort=True)
    cnt = grp.size()
    part = cnt[cnt < K]
    add("果数", int(cnt.size), n93, int(cnt.size) == n93, "对照值为 93 表c 的 n_fruit")
    dist = {int(k): int(v) for k, v in cnt.value_counts().sort_index().items()}
    add("配上光谱的面数", len(ap), n93 * K, "—",
        f"对照值为参考值表的面数 {n93} × {K}；每果面数的分布 {dist}，"
        f"未配上光谱的面 {n93 * K - len(ap)} 个，分属 {len(part)} 个果")
    n_multi = int(grp["y_fruit"].nunique().max())
    dev = float((grp["ssc"].mean() - grp["y_fruit"].first()).abs().max())
    add("y_fruit = 该果已配上光谱的面值的均值", dev, 0.0, dev < 1e-9 and n_multi == 1,
        f"全部 {cnt.size} 果的最大差（同一果内 y_fruit 取值数至多 {n_multi}）；缺面的 {len(part)} 果"
        "也是如此，所以 y_fruit 不是参考值表 5 个面的均值，93 的果间均方不能由它复算")

    # 两条必要条件：补上未配上光谱的面，只会让果内平方和与总平方和变大
    dev_in = ap["ssc"] - grp["ssc"].transform("mean")
    ssw_obs = float((dev_in ** 2).sum())
    ssw93 = n93 * (K - 1) * msw93
    add("果内平方和：93 表c ≥ 由已配面值算出的值", ssw93, ssw_obs, ssw93 >= ssw_obs,
        f"93 的值为 n(k−1)·MS_within；多出 {ssw93 - ssw_obs:.4f}")
    x = ap["ssc"].to_numpy(float)
    tss_obs = float(((x - x.mean()) ** 2).sum())
    tss93 = (n93 - 1) * msb93 + n93 * (K - 1) * msw93
    add("总平方和：93 表c ≥ 由已配面值算出的值", tss93, tss_obs, tss93 >= tss_obs,
        f"93 的值为 (n−1)·MS_between + n(k−1)·MS_within；多出 {tss93 - tss_obs:.4f}")

    # 五面齐全的果：参考值表里这些果的 5 个面都配上了光谱，可按同式重算，作为子集对照
    full = cnt[cnt == K].index
    mat: FloatArr = np.vstack([ap.loc[ap["id"] == fid, "ssc"].to_numpy(float) for fid in full])
    sub = anova(mat)
    ratio_sub = sub["var_face"] / sub["var_apple"]
    ms = [m_required(ratio_sub, r2)[1] for r2, _ in TARGETS]
    tag = f"五面齐全的 {len(full)} 果"
    add(f"{tag}：σ̂a²", sub["var_apple"], va93, "—", "子集对照，对照值为 633 果（93 表c）")
    add(f"{tag}：σ̂f²", sub["var_face"], vf93, "—", "子集对照，对照值为 633 果（93 表c）")
    add(f"{tag}：σ̂f²/σ̂a²", ratio_sub, ratio, "—", "子集对照，对照值为 633 果（93 表c）")
    moved = [f"R²max={r2} 一档为 {m}" for (r2, m_text), m in zip(TARGETS, ms, strict=True) if m != m_text]
    add(f"{tag}：设计表 m", "/".join(str(m) for m in ms), "/".join(str(t[1]) for t in TARGETS), "—",
        "子集对照；四档与 633 果相同" if not moved else
        f"子集对照；比值 {ratio_sub:.4f} 落在表b 所列区间之外，{'、'.join(moved)}")
    add(f"{tag}：√(σ̂a² + σ̂f²)", math.sqrt(sub["var_apple"] + sub["var_face"]), math.sqrt(va93 + vf93), "—",
        "子集对照，对照值为 633 果（93 表c）")
    return rows, x


def notes(ratio: float) -> list[dict[str, str]]:
    lo = max((m - 1) * (1 - r2) / r2 for r2, m in TARGETS)
    hi = min(m * (1 - r2) / r2 for r2, m in TARGETS)
    return [
        {"项": "本表回答的问题",
         "说明": "稿件 §2.5 写明正文报告的派生量各自落盘。设计表与单面取值 SD 由 93sscceiling_v18.xlsx "
                 "表c 的方差分量现算。此前设计表只作为稿件值的重放核对行出现在 A14reprocheck_f1_v18.xlsx 表a"
                 "（只有上取整后的 m）；两分量合成的单面取值 SD 未落盘，A14 表a 那一行是配上光谱的 3,149 个面的"
                 "样本 SD 2.207，记于稿件改用两分量口径之前。本表把两者连同算法写下来。"},
        {"项": "设计表",
         "说明": "式 (eq:design)：m ≥ (σf²/σa²)·R²max/(1−R²max)，取上整（与 A14reprocheck 同法，减 1e-9 "
                 "防浮点把整数抬过头）。表b 同时列出 m 处与 m−1 处的可达上界 1/(1+比值/m)，前者不低于 "
                 f"R²max、后者低于 R²max，即 m 是最小的够用值。比值落在 ({lo:.4f}, {hi:.4f}] 内时四档 m 都不变，"
                 f"本数据的比值 {ratio:.4f} 在其中；分产地的比值见 93 表c 的三行分产地方差分量。"},
        {"项": "单面取值 SD 的口径",
         "说明": "正文那句写的是单面取值的 SD 大于果内 SD，因为它还含果间分量 σ̂a，指的是两个分量合成的 "
                 "√(σ̂a²+σ̂f²)，与同段的 σ̂a²、σ̂f² 同源（633 果 × 5 面），读者可由正文所印的两个分量复算。"
                 "同一批面值的样本 SD 与只取配上光谱的面值的样本 SD 是另两种口径，表c 一并列出；"
                 "三者舍入到三位小数各不相同，正文用的是第一种。"},
        {"项": "parquet 核对的覆盖面",
         "说明": "v18_apple_faces.parquet 只有配上光谱的 3,149 个面；参考值表另有 16 个面没配上光谱，分属 12 个果，"
                 "这 12 个果的 y_fruit 是已配面值的均值，不是 5 面均值。所以 93 的两个均方不能由 parquet 逐位复算。"
                 "parquet 能核对的是：果数；两条必要条件（补上缺失的面只会让果内平方和与总平方和变大，93 的值都不低于"
                 "由已配面值算出的值）；以及五面齐全的 621 果按同式重算的方差分量、比值、设计表与 SD。最后一项是"
                 "子集对照，不要求与 633 果的值相同。方差分量本身以 93 表c 为准。"},
        {"项": "限度",
         "说明": "苹果糖度原始表不在服务器上，本脚本没有从原始表重算 93 的方差分量；它只读 93 表c、表e "
                 "的已存值，parquet 只用于核对上面写明的部分。"},
    ]


def main() -> int:
    xl = pd.ExcelFile(WB93)
    vc = pd.read_excel(xl, SHEET_VC).set_index("分组").loc["全体（主口径）"]
    ce = pd.read_excel(xl, SHEET_CEIL)
    var_truth_m1 = float(ce[(ce["分组"] == "全体") & (ce["m_faces_as_truth"] == 1)]["var_of_truth"].iloc[0])

    rows_a, rows_b, ratio = table_ab(vc)
    if os.path.isfile(APPLE):
        rows_d, ssc = parquet_checks(vc, ratio)
        rows_c = table_c(vc, var_truth_m1, ssc)
    else:
        rows_d = [{"核对项": "未核对", "parquet 实测": "—", "对照值": "—", "是否一致": "—",
                   "说明": f"找不到 {os.path.relpath(APPLE, ROOT)}"}]
        rows_c = table_c(vc, var_truth_m1, None)

    path = write_script_workbook(__file__, {
        0: ("方差分量与比值", pd.DataFrame(rows_a)),
        1: ("设计表", pd.DataFrame(rows_b)),
        2: ("单面取值SD三种口径", pd.DataFrame(rows_c)),
        3: ("与parquet核对", pd.DataFrame(rows_d)),
        4: ("读法与限度", pd.DataFrame(notes(ratio))),
    })
    print(f"写出 {path}")
    bad = 0
    for tag, rows in (("表a", rows_a), ("表b", rows_b), ("表c", rows_c), ("表d", rows_d)):
        for r in rows:
            flag = r["是否一致"]
            label = next(iter(r.values()))
            print(f"  {tag} {label}: {'✓' if flag is True else ('✗' if flag is False else '·')}")
            bad += int(flag is False)
    print(f"A17_MISMATCH {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
