"""C4 pilot · 第 4 段：001 苹果 2025 光谱侧的饱和体检（决定 C4 形态的硬约束）。

为什么必须做
------------
94/95 号脚本给出：按苹果分组留出时，PLS 预测整果均值 SSC 的 R² 只有 0.09–0.19
（新疆子集甚至为负）。在把这归因于"果实光学性质"或"文献虚高"之前，必须先排除
**采集端的技术故障**。本脚本查的就是这一条。

结论（先写在前面，正文据此定形态）
--------------------------------
光谱在**近红外要害区大面积贴顶饱和**，且饱和是系统性的、无干净子集：

  · 692–950 nm 区间整体贴 1；811 nm 处 **70.0%** 的光谱达到反射率 1.0
  · 394–567 nm 区间贴 0；411 nm 处 **52.2%** 的光谱低于 0.01
  · 346 个波段中 **246 个**饱和率 >1%，仅剩 100 个"干净"波段
  · 1,060 张采集图像中**没有一张**完全不饱和（最低仍有 4.0% 波段饱和）
  · 3,228 条光谱中完全不饱和的只有 **3 条**

⚠️ **但饱和并不是低预测性能的原因** —— 这一条必须写清楚，否则会得出错误归因：
  · 糖的 3 级 O–H/C–H 倍频带（约 900–980 nm）**平均贴 1 率仅 1.2%，是干净的**；
    饱和峰在 805.5 nm（71.4%），集中在 692–890 nm 的近红外平台区。
  · **只用干净波段建模反而更差**：全谱 R²=0.112 → 干净 100 波段 R²=0.069；
    单用 860–1001 nm R²=0.017；单用 900–1001 nm（糖带本身）**R²=0.012**。
  · 与 SSC 相关最强的单波段在 **678 nm（|r|=0.154）**，属叶绿素吸收/红边，
    即"色泽—成熟度"代理，而非糖的直接吸收。

真实原因（与物理一致，作为论文限制条款如实陈述）
--------------------------------------------
反射式几何在完整果实上的有效采样深度只有毫米量级，主要探到果皮与皮下组织；
而折光仪测的是破坏性取样后的**果肉**汁液，两者在空间上不是同一块组织。加之
391–1001 nm 只覆盖极弱的 3 级倍频，糖的强吸收带（2 级 ~1150–1250 nm、合频
~1400–1600 nm）全在量程之外。**因此本数据集光谱侧无法支撑近红外糖度定标建模，
这是采集配置的固有限制，不是预处理或波段选择能救的。**

对 C4 的后果（写进论文形态，不得回避）
------------------------------------
  · **参考值侧（SSC 的 5 面重复测定）完全不受影响** —— 方差分量、可达上界、
    1/m 设计公式、衰减比诊断，全部只用 SSC 数值，与光谱质量无关。这半边成立，
    且已被三个独立定量点预测验证（1/m 标度偏差 <2.3%；纯标签基线理论 0.3462 /
    实测 0.3456；衰减比理论 0.5816 / Formal 5 种子实测 0.5636，cluster CI95[0.531,0.596] 含理论值 —— 后者由
    A2forensicfix_v18.py 计算并落盘，订正了本 docstring 早前的 0.3464 与 0.515）。
  · **光谱侧（更好的估计器、序贯扫描决策）在本数据集上不可演示。**
  · 故 94 号脚本判出的"**分叉 A**"（局部光谱感知不到局部糖度）**必须撤回其一般性主张**：
    本数据集的光谱连**果水平**糖度都几乎感知不到（R²≈0.11），因此"感知不到面级偏离"
    不能推断为果实光学的普遍性质，只能陈述为本采集配置下的观测。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_DIR = os.path.join(DATA_ROOT, "001_apple_hyperspectral_multiyear")
HI, LO = 0.99, 0.01
NIR_LO, NIR_HI = 690.0, 960.0          # 糖倍频吸收所在区
SUGAR_LO, SUGAR_HI = 900.0, 980.0      # 3 级 O–H/C–H 倍频主带

logger = Logger(__file__)


def main() -> None:
    sp = pd.read_csv(os.path.join(DATA_DIR, "新疆-山东-甘肃苹果光谱数据2025.csv"))
    wc = [c for c in sp.columns if c.endswith("nm")]
    wl = np.array([float(c[:-2]) for c in wc])
    X = sp[wc].values.astype(float)
    sid = sp["实际苹果编号"].astype(str)
    logger.log(f"光谱 {X.shape}，{wl.min():.1f}–{wl.max():.1f} nm")

    # ---------- 表a：逐波段饱和率 ----------
    lo_r = (X < LO).mean(0) * 100
    hi_r = (X > HI).mean(0) * 100
    tab_band = pd.DataFrame({
        "波长nm": wl, "贴0率%": lo_r, "贴1率%": hi_r,
        "总饱和率%": lo_r + hi_r,
        "均值反射率": X.mean(0), "SD": X.std(0, ddof=1),
        "是否坏带(>1%)": (lo_r + hi_r) > 1,
    })

    # ---------- 表b：坏带区段与可用波段 ----------
    bad = (lo_r + hi_r) > 1
    segs, st = [], None
    for i, b in enumerate(bad):
        if b and st is None:
            st = i
        if not b and st is not None:
            segs.append((wl[st], wl[i - 1], i - st)); st = None
    if st is not None:
        segs.append((wl[st], wl[-1], len(bad) - st))
    tab_seg = pd.DataFrame(segs, columns=["起nm", "止nm", "波段数"])
    tab_seg["类型"] = ["贴0（信号不足）" if wl[np.argmin(np.abs(wl - s))] < 600
                     else "贴1（过曝饱和）" for s, _, _ in segs]

    sugar = (wl >= SUGAR_LO) & (wl <= SUGAR_HI)
    tab_summary = pd.DataFrame([
        {"项": "总波段数", "值": len(wl)},
        {"项": "坏带数（饱和率>1%）", "值": int(bad.sum())},
        {"项": "干净波段数", "值": int((~bad).sum())},
        {"项": f"糖倍频带 {SUGAR_LO:.0f}–{SUGAR_HI:.0f}nm 内波段数", "值": int(sugar.sum())},
        {"项": "其中坏带数", "值": int((bad & sugar).sum())},
        {"项": "糖倍频带内平均贴1率(%)", "值": float(hi_r[sugar].mean())},
        {"项": "最高贴1率(%)", "值": float(hi_r.max())},
        {"项": "最高贴1率对应波长(nm)", "值": float(wl[int(np.argmax(hi_r))])},
        {"项": "最高贴0率(%)", "值": float(lo_r.max())},
        {"项": "最高贴0率对应波长(nm)", "值": float(wl[int(np.argmax(lo_r))])},
    ])

    # ---------- 表c：逐条光谱 / 逐图像 / 逐产地 的饱和分布 ----------
    nir = (wl >= NIR_LO) & (wl <= NIR_HI)
    sat = (X[:, nir] > HI).mean(1) * 100
    per_img = pd.DataFrame({"img": sp["图像名称"].astype(str), "sat": sat}).groupby("img")["sat"].mean()
    rows = [
        {"层级": "逐条光谱", "n": len(sat), "饱和率均值%": sat.mean(),
         "中位%": np.median(sat), "最小%": sat.min(), "最大%": sat.max(),
         "完全不饱和数": int((sat == 0).sum())},
        {"层级": "逐采集图像", "n": len(per_img), "饱和率均值%": per_img.mean(),
         "中位%": per_img.median(), "最小%": per_img.min(), "最大%": per_img.max(),
         "完全不饱和数": int((per_img == 0).sum())},
    ]
    org = sid.str[:2]
    for g in sorted(org.unique()):
        s = sat[(org == g).values]
        rows.append({"层级": f"产地 {g}", "n": len(s), "饱和率均值%": s.mean(),
                     "中位%": np.median(s), "最小%": s.min(), "最大%": s.max(),
                     "完全不饱和数": int((s == 0).sum())})
    tab_level = pd.DataFrame(rows)

    # ---------- 表d：可用子集有多大 ----------
    tab_sub = pd.DataFrame([
        {"阈值：NIR饱和率≤": f"{t}%", "光谱条数": int((sat <= t).sum()),
         "占比%": 100 * (sat <= t).mean(),
         "涉及苹果数": sid[sat <= t].str.rsplit("-", n=1).str[0].nunique()}
        for t in (0, 1, 5, 10, 20)
    ])

    out = write_script_workbook(__file__, {
        0: ("逐波段饱和率", tab_band),
        1: ("坏带区段", tab_seg),
        2: ("汇总", tab_summary),
        3: ("逐条/逐图像/逐产地饱和分布", tab_level),
        4: ("可用子集规模", tab_sub),
    })

    logger.log(f"坏带 {int(bad.sum())}/{len(wl)}，干净波段仅 {int((~bad).sum())}")
    logger.log(f"最高贴1率 {hi_r.max():.1f}% @ {wl[int(np.argmax(hi_r))]:.1f}nm")
    logger.log(f"糖倍频带 {SUGAR_LO:.0f}–{SUGAR_HI:.0f}nm 平均贴1率 {hi_r[sugar].mean():.1f}%")
    logger.log(f"完全不饱和图像 {int((per_img==0).sum())}/{len(per_img)}；"
               f"完全不饱和光谱 {int((sat==0).sum())}/{len(sat)}")
    logger.log("→ 结论：光谱侧不可用于近红外糖度定标；参考值侧不受影响。")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
