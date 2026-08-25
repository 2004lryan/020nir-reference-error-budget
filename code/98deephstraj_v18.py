"""C3 pilot · 第 1 段：DeepHS 逐果成熟轨迹的提取与「信号是否存在」判定。

背景
----
C3 的主张：果实后熟在光谱空间中沿一条**共同路径**演进，因此单次扫描可反演「生物龄」
（而非日历天数）。本脚本先不做任何建模，只回答一件事：**这条路径的信号在不在**。

数据现实（已核验，`01pipeline_stage.md` ②ter）
-------------------------------------------
· 物理实体 = (fruit, sample_no)，共 **257** 个，其中 **225** 个有 ≥3 天成像。
  （早前记录的「804 个实体」是错的——那是 (fruit,sample,camera,side) 复合键，
    把同一颗果的 front/back × 三种相机各算成了独立实体，多算 3.6 倍。）
· **破坏性标签每果基本只有 1 个时间点**（253 个有标注实体里 222 个只覆盖 1 天），
  故无法学「光谱轨迹 → 成熟度轨迹」，每条轨迹只有一个锚点。
· 光谱质量好：抽样零饱和波段，反射率 0.04–0.89。

本脚本的关键设计：用 front/back 作采集噪声的技术重复
------------------------------------------------
同一颗果在**同一天**的 front 与 back 是两次独立采集（重新摆位、重新成像）。因此
  · 同日 front↔back 的差异  ≈ 采集噪声（摆位/光照/朝向）
  · 跨日同侧的差异          ≈ 采集噪声 + 成熟信号
两者相除即可判定"日间变化到底是成熟信号还是采集噪声"，**不需要假设**。
这是 001 苹果那批数据没有的结构，也是本轮必须先用掉的证据。

预注册 kill criterion（先写后跑，不得事后调整）
--------------------------------------------
  kill ①「无可提取的成熟信号」：
        果内跨日方差 / 同日 front-back 方差 的中位数 < 1.5
  kill ②「无共同单调路径」：
        轨迹 FPCA 的 PC1 得分与成像日序的 Spearman |ρ| 中位数 < 0.5
  kill ③「单次扫描不能测龄」：
        按果分组留出、单条谱预测该谱所在日序，R² < 0.3
任一条触发即如实记录并停止 C3 立项，不放宽、不换指标。
本脚本只产出 ① 所需证据与轨迹数据集；②③ 由 99 号脚本消费本脚本的产物。

产物
----
`03data/processed/v18_deephs_traj.parquet` —— 逐 (fruit, sample, camera, side, day)
的 ROI 均值谱。落盘后即置为只读，后续步骤一律不得改写。
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEED = 2004
import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
ROOT = (os.path.join(DATA_ROOT, "003_deephs_fruit_hyperspectral") + "/"
        "DeepHS-Fruit-2023-Datasets")
ANN = os.path.join(ROOT, "annotations")
OUT_PARQUET = "03data/processed/v18_deephs_traj.parquet"

CAMERA = "VIS"          # 5 个物种都有 VIS；NIR 只有 Avocado/Kiwi
SPECULAR_PCT = 99.0     # 高于该分位的像素判为镜面高光，剔除
BG_PCT = 60.0           # ROI 阈值分位（背景暗、果实亮）
KILL1_RATIO = 1.5       # 预注册

logger = Logger(__file__)


def read_hdr(p: str) -> dict[str, str]:
    d = {}
    for ln in open(p, errors="replace"):
        if "=" in ln:
            k, v = ln.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def roi_mean(binp: str) -> tuple[np.ndarray, int] | None:
    """取果实主体像素的均值谱。返回 (谱, ROI像素数)。"""
    hp = binp[:-4] + ".hdr"
    if not os.path.exists(hp):
        return None
    h = read_hdr(hp)
    try:
        S, L, B = int(h["samples"]), int(h["lines"]), int(h["bands"])
    except KeyError:
        return None
    try:
        cube = np.asarray(np.memmap(binp, dtype="<f4", mode="r", shape=(L, S, B)))
    except ValueError:
        return None
    flat = cube.reshape(-1, B)
    ref = flat[:, B // 2]                                  # 中段波段做亮度参考
    thr = np.percentile(ref, BG_PCT)
    hi = np.percentile(ref, SPECULAR_PCT)
    m = (ref > thr) & (ref <= hi)                          # 去背景 + 去镜面高光
    if m.sum() < 200:
        return None
    return flat[m].mean(0).astype(np.float32), int(m.sum())


def load_records() -> tuple[list[dict], dict[int, dict], dict[str, list[float]]]:
    recs, anns = [], []
    for f in ("train_all_v2.json", "val_v2.json", "test_v2.json"):
        d = json.load(open(os.path.join(ANN, f)))
        recs += d["records"]
        anns += d["annotations"]
        cams = {c["id"]: c["wavelengths"] for c in d["cameras"]}
    ann_by_rec: dict[int, dict] = {}
    for a in anns:
        ann_by_rec.setdefault(a["record_id"], a)
    return recs, ann_by_rec, cams


def sample_of(r: dict) -> tuple[str, str | None]:
    b = os.path.basename(r["files"]["data_file"])[:-4]
    rest = b[len(f"{r['fruit'].lower()}_{r['day']}_"):]
    for s in ("_front", "_back"):
        if rest.endswith(s):
            return rest[: -len(s)], s[1:]
    return rest, None


def main() -> None:
    recs, ann_by_rec, cams = load_records()
    wl = np.array(cams[CAMERA], dtype=float)
    sel = [r for r in recs if r["camera_type"] == CAMERA]
    logger.log(f"[98deephstraj] camera={CAMERA} 记录 {len(sel)} 条，波段 {len(wl)} "
               f"({wl.min():.1f}–{wl.max():.1f} nm)")

    rows, spectra = [], []
    t0 = time.time()
    miss = 0
    for i, r in enumerate(sel):
        binp = os.path.join(ROOT, r["files"]["data_file"])
        if not os.path.exists(binp):
            miss += 1
            continue
        got = roi_mean(binp)
        if got is None:
            miss += 1
            continue
        mu, npix = got
        if len(mu) != len(wl):
            miss += 1
            continue
        s, side = sample_of(r)
        a = ann_by_rec.get(r["id"], {})
        rows.append({
            "fruit": r["fruit"], "sample": s, "side": side, "day": r["day"],
            "camera": CAMERA, "record_id": r["id"], "n_roi_px": npix,
            "storage_days": a.get("storage_days"),
            "firmness": a.get("firmness"),
            "ripeness_state": a.get("ripeness_state"),
        })
        spectra.append(mu)
        if (i + 1) % 300 == 0:
            logger.log(f"  {i+1}/{len(sel)}  用时 {time.time()-t0:.0f}s")
    logger.log(f"提取完成 {len(rows)} 条，缺失/跳过 {miss}，用时 {time.time()-t0:.0f}s")

    X = np.vstack(spectra)
    meta = pd.DataFrame(rows)
    spec_df = pd.DataFrame(X, columns=[f"w{w:.2f}" for w in wl])
    df = pd.concat([meta.reset_index(drop=True), spec_df], axis=1)

    # day 序号：day_01..day_NN 与 day_mK_NN 两套命名，各自排序后给全局序
    day_order = {d: i for i, d in enumerate(sorted(meta["day"].unique()))}
    df["day_idx"] = df["day"].map(day_order)

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    logger.log(f"轨迹数据集 → {OUT_PARQUET}  shape={df.shape}")

    # ---------- kill ①：跨日方差 / 同日 front-back 方差 ----------
    wcols = [c for c in df.columns if c.startswith("w")]
    same_day, cross_day = [], []
    for (fr, sm), g in df.groupby(["fruit", "sample"]):
        # 同日 front vs back：采集噪声
        for d, gd in g.groupby("day"):
            if gd["side"].nunique() == 2 and len(gd) >= 2:
                f_ = gd[gd["side"] == "front"][wcols].to_numpy(float).mean(0)
                b_ = gd[gd["side"] == "back"][wcols].to_numpy(float).mean(0)
                same_day.append(float(np.mean((f_ - b_) ** 2) / 2))
        # 跨日同侧：采集噪声 + 成熟信号
        for side, gs in g.groupby("side"):
            if gs["day"].nunique() >= 2:
                M = gs.groupby("day")[wcols].mean().to_numpy(float)
                cross_day.append(float(M.var(axis=0, ddof=1).mean()))
    sd_arr, cd_arr = np.array(same_day), np.array(cross_day)
    ratio = float(np.median(cd_arr) / np.median(sd_arr)) if len(sd_arr) else np.nan
    verdict1 = ("触发 kill ①（日间变化被采集噪声主导，无可提取成熟信号）"
                if ratio < KILL1_RATIO else "通过 kill ①（日间变化显著超出采集噪声）")

    tab_kill1 = pd.DataFrame([
        {"项": "同日 front↔back 方差（采集噪声）中位", "值": float(np.median(sd_arr)),
         "n": len(sd_arr)},
        {"项": "跨日同侧方差（噪声+成熟信号）中位", "值": float(np.median(cd_arr)),
         "n": len(cd_arr)},
        {"项": "比值（跨日 / 同日）", "值": ratio, "n": np.nan},
        {"项": f"预注册阈值 kill ① （比值 < {KILL1_RATIO} 即触发）", "值": KILL1_RATIO,
         "n": np.nan},
        {"项": "**判定**", "值": verdict1, "n": np.nan},
    ])

    ent = df.groupby(["fruit", "sample"])["day"].nunique()
    tab_struct = pd.DataFrame([
        {"项": "提取成功的谱条数", "值": len(df)},
        {"项": "物理实体数 (fruit,sample)", "值": int(len(ent))},
        {"项": "≥3 天成像的实体数", "值": int((ent >= 3).sum())},
        {"项": "同时有 front 与 back 的 (实体,日) 单元数", "值": len(sd_arr)},
        {"项": "有 storage_days 的谱条数", "值": int(df["storage_days"].notna().sum())},
        {"项": "有 firmness 的谱条数", "值": int(df["firmness"].notna().sum())},
        {"项": "ROI 像素数 中位", "值": float(df["n_roi_px"].median())},
        {"项": "反射率范围", "值": f"[{X.min():.3f}, {X.max():.3f}]"},
        {"项": "饱和格子占比%（>0.99 或 <0.01）",
         "值": float(100 * np.mean((X > 0.99) | (X < 0.01)))},
    ])

    out = write_script_workbook(__file__, {
        0: ("数据结构与光谱质量", tab_struct),
        1: ("kill① 成熟信号 vs 采集噪声", tab_kill1),
    })
    logger.log(f"\n=== kill ① 判定 ===")
    logger.log(f"同日 front-back 方差中位 = {np.median(sd_arr):.6g} (n={len(sd_arr)})")
    logger.log(f"跨日同侧方差中位       = {np.median(cd_arr):.6g} (n={len(cd_arr)})")
    logger.log(f"比值 = {ratio:.3f}   预注册阈值 = {KILL1_RATIO}")
    logger.log(verdict1)
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
