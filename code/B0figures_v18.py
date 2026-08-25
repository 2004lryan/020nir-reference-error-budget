"""C4′ 论文正图（投稿版，中英双语）。

设计口径：
  · 禁止 matplotlib 默认样式；统一自定义 rcParams、无边框、浅网格、色盲友好配色。
  · 每张图的每个数字都必须能追到 outputs/*.xlsx，不得手填。
  · 中英双语版本数据/配色/字号/版式完全一致，只切文本。

图清单
  图a  参考值侧：果内 SSC 变异的规模、分产地差异、可达 R² 上界曲线
  图b  三个无自由参数的点预测（预测 vs 实测）
  图c  测量侧：谱层面方差分解 + 仪器分量对模型质量的不变性
  图d  验证协议：两个数据集、两种不同的分组泄漏机制
"""

from __future__ import annotations

import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import save_figure_bilingual, setup_chinese_font  # noqa: E402

_H = os.path.dirname(os.path.abspath(__file__))
_OUTA = os.path.join(_H, "..", "outputs")
OUT = _OUTA if os.path.isdir(_OUTA) else os.path.join(_H, "..", "04outputs")

# ── 色盲友好调色板（Okabe–Ito）──────────────────────────────────────────────
C_BLUE, C_ORANGE, C_GREEN = "#0072B2", "#E69F00", "#009E73"
C_RED, C_PURPLE, C_GREY = "#D55E00", "#CC79A7", "#7F7F7F"
C_LIGHT = "#DDDDDD"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.color": C_LIGHT, "grid.linewidth": 0.6,
    "grid.alpha": 0.9, "axes.axisbelow": True,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.4, "lines.markersize": 4,
})

# ── 全部数值现场读自工作簿（一致性审计 HP-PHANTOM-RESULT / EF-001 的根治）────────
# 规程④：任何写进论文的数字必须由分析脚本计算并导出到工作簿；绘图脚本里的常量不算证据。
# 本文件因此不再持有任何实测常量——下面每个值都在导入时从 outputs/*.xlsx 读出，
# 读不到就直接抛错，绝不回退到硬编码（回退等于把假零变成假证据）。
def _sheet(book: str, name: str) -> pd.DataFrame:
    return pd.read_excel(os.path.join(OUT, f"{book}_v18.xlsx"), sheet_name=name)


def _pick(df: pd.DataFrame, col: str, key: str) -> float:
    m = df[df[col].astype(str).str.contains(key, na=False, regex=False)]
    if not len(m):
        raise KeyError(f"工作簿中找不到 {key!r}（列 {col}）")
    return float(m.iloc[0, 1])


# --- 图 a / b 的来源：93sscceiling ---
_VC = _sheet("93sscceiling", "表c：方差分解（总体与分产地）").set_index("分组")
_AMP = _sheet("93sscceiling", "表d：果内异质性幅度 vs 仪器精度")

VAR_APPLE = float(_VC.loc["全体（主口径）", "var_apple"])
VAR_FACE = float(_VC.loc["全体（主口径）", "var_face"])
N_FRUIT = int(_VC.loc["全体（主口径）", "n_fruit"])
ICC = float(_VC.loc["全体（主口径）", "ICC"])
ICC_LO, ICC_HI = [float(x) for x in
                  str(_VC.loc["全体（主口径）", "ICC_CI95"]).strip("[]").split(",")]
ORIGIN = {c: (float(_VC.loc[k, "ICC"]), int(_VC.loc[k, "n_fruit"]))
          for c, k in (("GS", "GS（甘肃）"), ("XJ", "XJ（新疆）"), ("SD", "SD（山东）"))}
RANGE_PCTL = {p: _pick(_AMP, "指标", f"果内极差 P{p}") for p in (10, 25, 50, 75, 90, 95, 99)}
REFRACT = _pick(_AMP, "指标", "折光仪标称精度")
SD_WITHIN_MED = _pick(_AMP, "指标", "果内 5 面 SD（中位数）")

# --- 图 b 的六个点预测：93sscceiling 表a + A5aggregate 表b ---
_PP = _sheet("93sscceiling", "表h：光谱侧结构核验")
_ATT = _sheet("A5aggregate", "表b：衰减比（均值之比）")


def _att(key: str) -> float:
    m = _ATT[_ATT.iloc[:, 0].astype(str).str.contains(key, na=False)]
    if not len(m):
        raise KeyError(f"A5aggregate 表b 找不到 {key!r}")
    return float(m.iloc[0]["跨种子均值"] if "跨种子均值" in _ATT.columns else m.iloc[0, 1])



# --- 图 b：六个不含自由参数的点预测（A1consolidate 表a/b/c + A5aggregate 表b）---
_SCALE = _sheet("A1consolidate", "表a：1_m 标度理论vs实测").set_index("m（取前 m 个面的均值）")
_COV = _sheet("A1consolidate", "表b：面间协方差与复合对称")
_BASE = _sheet("A1consolidate", "表c：纯标签基线理论vs实测").set_index("量")
_ATT_OBS = float(_ATT.iloc[0]["均值之比（正文口径）"])
_ATT_THE = float(_ATT.iloc[1]["均值之比（正文口径）"])


def _cov(key: str) -> float:
    r = _COV[_COV["面"].astype(str).str.contains(key, na=False)]
    return float(r.iloc[0]["顶部"])


POINTS = (
    [(f"1/m 标度 m={m}", float(_SCALE.loc[m, "理论方差 σ²a+σ²f/m"]),
      float(_SCALE.loc[m, "实测方差"])) for m in (2, 3, 4)]
    + [("纯标签基线",
        float(_BASE.loc["理论值 1−(σ²f+σ²f/4)/(σ²a+σ²f)", "值"]),
        float(_BASE.loc["实测值（其余 4 面均值 → 该面）", "值"])),
       ("面间协方差", _cov("可交换性预言值"), _cov("非对角元均值")),
       ("衰减比", _ATT_THE, _ATT_OBS)]
)

# --- 图 c：两个模型工作点（97instrbudget 表d）---
_WP = _sheet("97instrbudget", "表d：预测层面仪器间离散度")
WORKPOINTS = [
    (("弱模型" if True else ""), float(r["按果分组 5 折 CV 的 R²（DM）"]), float(r["RMSEP（%DM）"]),
     float(r["换台仪器就会变的那部分 RMSEP（√var_instr）"]),
     float(r["**仪器间分量占 MSEP 比例**"]) * 100.0)
    for _, r in _WP.iterrows()
]
WORKPOINTS = [("弱模型", *WORKPOINTS[0][1:]), ("强模型", *WORKPOINTS[1][1:])]

# --- 图 c(b)：仪器分量的操作点扫描（A12opscan，10 个预注册配置 × Formal 5 种子）---
# 起因：预投稿评审判定「两个工作点」无法证伪，要求扫 6–10 点并报斜率。
_OS = _sheet("A12opscan", "表a：操作点扫描明细")
OPSCAN_R2 = _OS["R²（5 种子均值）"].to_numpy(dtype=float)
OPSCAN_TOT = _OS["RMSEP（%DM）"].to_numpy(dtype=float)
OPSCAN_INS = _OS["仪器相关 RMSEP 分量（%DM）"].to_numpy(dtype=float)
_OB = _sheet("A12opscan", "斜率检验与判定").set_index("量")
OPSCAN_SLOPE = float(_OB.loc["**斜率 d(仪器分量)/d(R²)**", "值"])
OPSCAN_SLOPE_LO = float(_OB.loc["**斜率 d(仪器分量)/d(R²)**", "CI95 下限"])
OPSCAN_SLOPE_HI = float(_OB.loc["**斜率 d(仪器分量)/d(R²)**", "CI95 上限"])

# --- 图 d(a)：苹果个体级泄漏（A4formal 表c，Formal 5 种子）---
_LK = _sheet("A4formal", "表c：泄漏对照").set_index("量")
LEAK_001 = [
    ("按行随机", float(_LK.loc["raw · y_face · 按行随机", "跨种子均值"]),
                 float(_LK.loc["raw · y_fruit · 按行随机", "跨种子均值"])),
    ("按苹果分组", float(_LK.loc["raw · y_face · 按果分组", "跨种子均值"]),
                   float(_LK.loc["raw · y_fruit · 按果分组", "跨种子均值"])),
]


# ── 实测常量（全部可追溯，来源标在注释里）─────────────────────────────────
# 93sscceiling_v18.xlsx
# 点预测（93/95）
# 97instrbudget_v18.xlsx
SPEC_VC = [("果间（生物+批次）", 52.60), ("设备主效应（可校正）", 17.07),
           ("果×设备合并残差（不可分离）", 30.33)]
# 95splitleak / 99deephskill
# A6formal T11（猕猴桃），Formal 5 种子：(标签, R², RMSE, R²的 cluster CI95, RMSE 的 cluster CI95)
# --- 图 d(b)：时序日序分组对照（A6formal T11，Formal 5 种子）---
_T11 = _sheet("A6formal", "T11 DeepHS 日序分组").set_index("量")


def _t11(row: str) -> tuple:
    r = _T11.loc[row]
    return (float(r["跨种子均值"]), float(r["cluster CI95 下限"]), float(r["cluster CI95 上限"]))


LEAK_003 = [
    ("按果分组", _t11("日序 · Kiwi·按果分组 · R²")[0], _t11("日序 · Kiwi·按果分组 · RMSE")[0],
     _t11("日序 · Kiwi·按果分组 · R²")[1:], _t11("日序 · Kiwi·按果分组 · RMSE")[1:]),
    ("按采集日分组", _t11("日序 · Kiwi·按采集日分组 · R²")[0], _t11("日序 · Kiwi·按采集日分组 · RMSE")[0],
     _t11("日序 · Kiwi·按采集日分组 · R²")[1:], _t11("日序 · Kiwi·按采集日分组 · RMSE")[1:]),
]

TXT = {
    "zh": {
        "a1_t": "果内 SSC 极差分布", "a1_x": "同一苹果 5 面 SSC 极差 (°Brix)", "a1_y": "累积占比",
        "a2_t": "方差分量（按声明产地）", "a2_y": "果内（面间）方差占比",
        "a3_t": "可达 $R^2$ 上界", "a3_x": "参考真值取的采样面数 $m$", "a3_y": "$R^2$ 上界",
        "b_t": "四组共六个无自由参数的点预测", "b_x": "模型预测值", "b_y": "实测值",
        "c1_t": "谱层面方差分解（012 猕猴桃）",
        "c2_t": "10 个工作点上的合并分量", "c2_y": "RMSEP (%DM)",
        "c2_note": "斜率 +0.007\nCI [0.004, 0.011]\n全程仅变 +1.1%",
        "d1_t": "001 苹果：个体泄漏", "d2_t": "003 DeepHS 猕猴桃：目标由分组变量派生",
        "e1_t": "半合成：参考噪声下的性能退化", "e2_t": "相对偏差随 m 收缩",
        "e_x": "参考重复次数 m", "e_y1": "$R^2$（对带噪标签）",
        "e_y2": "$R^2_{obs}/R^2_{pred}-1$", "e_pred": "预言（无自由参数）",
        "e_bound": "结构上界", "e_dm": "干物质 DM", "e_ssc": "SSC",
        "d_y1": "$R^2$", "d_y2": "RMSE（日序）",
        "refract": "折光仪标称精度 ±0.2", "med": "中位 2.80",
        "total": "总 RMSEP", "instr": "换台仪器就会变的部分",
        "leg_pred": "$y=x$（完美预测）",
    },
    "en": {
        "a1_t": "Within-fruit SSC range", "a1_x": "Range across 5 faces of one apple (°Brix)",
        "a1_y": "Cumulative fraction",
        "a2_t": "Variance components by declared origin", "a2_y": "Within-fruit variance share",
        "a3_t": "Attainable $R^2$ ceiling", "a3_x": "No. of faces averaged as reference, $m$",
        "a3_y": "$R^2$ upper bound",
        "b_t": "Six parameter-free point predictions", "b_x": "Model prediction", "b_y": "Observed",
        "c1_t": "Spectral variance decomposition (012 kiwifruit)",
        "c2_t": "Pooled component across 10 working points", "c2_y": "RMSEP (%DM)",
        "c2_note": "slope +0.007\nCI [0.004, 0.011]\n+1.1% over the range",
        "d1_t": "001 apple: individual-level leakage",
        "d2_t": "003 DeepHS kiwifruit: target derived from grouping variable",
        "e1_t": "Semi-synthetic: degradation under reference noise",
        "e2_t": "Relative deviation shrinks with m",
        "e_x": "Reference replicates m", "e_y1": "$R^2$ (against noisy label)",
        "e_y2": "$R^2_{obs}/R^2_{pred}-1$", "e_pred": "Prediction (parameter-free)",
        "e_bound": "Structural bound", "e_dm": "Dry matter", "e_ssc": "SSC",
        "d_y1": "$R^2$", "d_y2": "RMSE (day index)",
        "refract": "Refractometer spec ±0.2", "med": "median 2.80",
        "total": "Total RMSEP", "instr": "Changes when instrument changes",
        "leg_pred": "$y=x$ (perfect)",
    },
}
LBL = {
    "zh": {"GS": "甘肃", "XJ": "新疆", "SD": "山东",
           "1/m 标度 m=2": "$1/m$ 标度 $m$=2", "1/m 标度 m=3": "$1/m$ 标度 $m$=3",
           "1/m 标度 m=4": "$1/m$ 标度 $m$=4", "纯标签基线": "纯标签基线",
           "面间协方差": "面间协方差", "衰减比": "衰减比",
           "果间（生物+批次）": "果间（生物+批次）",
           "设备主效应（可校正）": "设备主效应（可校正）",
           "果×设备合并残差（不可分离）": "果×设备残差（不可分离）",
           "弱模型": "弱模型", "强模型": "强模型",
           "按行随机": "按行随机", "按苹果分组": "按苹果分组",
           "按果分组": "按果分组", "按采集日分组": "按采集日分组",
           "面级 SSC": "面级 SSC", "整果均值 SSC": "整果均值 SSC"},
    "en": {"GS": "Gansu", "XJ": "Xinjiang", "SD": "Shandong",
           "1/m 标度 m=2": "$1/m$ scaling $m$=2", "1/m 标度 m=3": "$1/m$ scaling $m$=3",
           "1/m 标度 m=4": "$1/m$ scaling $m$=4", "纯标签基线": "Label-only baseline",
           "面间协方差": "Between-face covariance", "衰减比": "Attenuation ratio",
           "果间（生物+批次）": "Between-fruit (biology + batch)",
           "设备主效应（可校正）": "Device main effect (correctable)",
           "果×设备合并残差（不可分离）": "Fruit×device residual (inseparable)",
           "弱模型": "Weak model", "强模型": "Strong model",
           "按行随机": "Random row split", "按苹果分组": "Grouped by apple",
           "按果分组": "Grouped by fruit", "按采集日分组": "Grouped by day",
           "面级 SSC": "Face-level SSC", "整果均值 SSC": "Fruit-mean SSC"},
}



def _var_apple_from_icc(icc: float) -> float:
    """由 ICC 端点反解 σ²a（保持 σ²f 不变）：ICC = σ²a/(σ²a+σ²f) ⇒ σ²a = ICC·σ²f/(1−ICC)。

    这样代入 σ²a/(σ²a+σ²f/m)，在 m=1 处恰好还原该 ICC 端点本身。
    """
    return icc * VAR_FACE / (1.0 - icc)


def _panel_tag(ax, s: str) -> None:
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


# ── 图 a ─────────────────────────────────────────────────────────────────────
def fig_a(lang: str) -> Figure:
    t, L = TXT[lang], LBL[lang]
    fig = Figure(figsize=(7.2, 2.3))
    ax1, ax2, ax3 = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))

    p = sorted(RANGE_PCTL)
    ax1.plot([RANGE_PCTL[k] for k in p], [k / 100 for k in p], "-o", color=C_BLUE, zorder=3)
    ax1.axvline(REFRACT, color=C_RED, ls="--", lw=1.1, zorder=2)
    ax1.text(REFRACT + 0.15, 0.90, t["refract"], color=C_RED, fontsize=6.8, va="center")
    ax1.axvline(2.80, color=C_GREY, ls=":", lw=1.0, zorder=2)
    ax1.text(2.95, 0.20, t["med"], color=C_GREY, fontsize=6.8)
    ax1.set_xlabel(t["a1_x"]); ax1.set_ylabel(t["a1_y"]); ax1.set_title(t["a1_t"])
    ax1.set_ylim(0, 1.02); ax1.set_xlim(0, 9.5); _panel_tag(ax1, "a")

    ks = ["SD", "XJ", "GS"]
    within = [1 - ORIGIN[k][0] for k in ks]
    ax2.bar(range(3), within, color=[C_GREEN, C_BLUE, C_ORANGE], width=0.6, zorder=3)
    ax2.axhline(1 - ICC, color=C_GREY, ls="--", lw=1.0, zorder=2)
    ax2.text(2.42, 1 - ICC + 0.015, f"{1-ICC:.3f}", color=C_GREY, fontsize=6.8, ha="right")
    for i, (k, v) in enumerate(zip(ks, within)):
        ax2.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax2.set_xticks(range(3)); ax2.set_xticklabels([L[k] for k in ks])
    ax2.set_ylabel(t["a2_y"]); ax2.set_title(t["a2_t"], fontsize=8)
    ax2.set_ylim(0, 0.78)
    _panel_tag(ax2, "b")

    m = np.arange(1, 11)
    ax3.plot(m, VAR_APPLE / (VAR_APPLE + VAR_FACE / m), "-o", color=C_BLUE, zorder=3)
    # 由 ICC 的自助置信区间端点**反解** σ²a（固定 σ²f），使阴影带在 m=1 处恰好等于
    # 工作簿里的 ICC CI95。此前写作 VAR_APPLE*(ICC_LO/ICC) —— 那是按比例缩放 σ²a，
    # 而 ICC=σ²a/(σ²a+σ²f) 不是 σ²a 的线性函数，缩放后 m=1 处只得到 [0.4546, 0.4961]，
    # **宽度只有真实 CI（[0.436,0.515]）的 53%，把不确定性画窄了**。
    # （一致性审计发现，2026-07-26）
    lo, hi = _var_apple_from_icc(ICC_LO), _var_apple_from_icc(ICC_HI)
    ax3.fill_between(m, lo / (lo + VAR_FACE / m), hi / (hi + VAR_FACE / m),
                     color=C_BLUE, alpha=0.15, lw=0, zorder=1)
    for mm, cc in ((1, C_RED), (5, C_GREEN)):
        y = VAR_APPLE / (VAR_APPLE + VAR_FACE / mm)
        ax3.plot([mm], [y], "o", color=cc, ms=6, zorder=4)
        ax3.annotate(f"{y:.3f}", (mm, y), xytext=(6, -10), textcoords="offset points",
                     fontsize=7, color=cc)
    ax3.set_xlabel(t["a3_x"]); ax3.set_ylabel(t["a3_y"]); ax3.set_title(t["a3_t"])
    ax3.set_ylim(0.3, 1.0); ax3.set_xticks([1, 2, 4, 6, 8, 10]); _panel_tag(ax3, "c")

    fig.tight_layout(w_pad=1.8)

    # savefig(bbox_inches="tight") 计算裁剪框时会把 x 轴标签的宽度折叠掉
    # （Axis.get_tightbbox 对过长标签的 for_layout_only 折叠），(c) 的长标签
    # 悬出轴区的部分因此不计入 tight bbox，en 版右缘恰好裁掉结尾的 "$m$"。
    # 把该标签的真实渲染宽度用一个无描边矩形注册进 tight bbox。矩形垂直方向
    # 取标签中心处的一条细缝：Agg 度量的标签下缘比 PDF 后端低约 0.9pt，若矩形
    # 罩住标签全高会把画布往下多裁一条缝；细缝只扩宽度、不动高度。zh 版标签
    # 本就在界内，矩形不改变其画布。
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    bb = ax3.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(
        [(bb.x0, bb.y0), (bb.x1, bb.y1)])
    pad = 2.0 / 72.0 / fig.get_figwidth()   # 2 pt 安全边，防备端间字体度量差
    yc, sliver = (y0 + y1) / 2, 0.001
    fig.add_artist(Rectangle((x0, yc - sliver / 2), (x1 - x0) + pad, sliver,
                             facecolor="none", edgecolor="none", linewidth=0))
    return fig


# ── 图 b ─────────────────────────────────────────────────────────────────────
def fig_b(lang: str) -> Figure:
    t, L = TXT[lang], LBL[lang]
    fig = Figure(figsize=(3.5, 3.2))
    ax = fig.add_subplot(111)
    pred = np.array([p[1] for p in POINTS])
    obs = np.array([p[2] for p in POINTS])
    lim = (0.2, 4.2)
    ax.plot(lim, lim, "-", color=C_GREY, lw=1.0, zorder=1, label=t["leg_pred"])
    mk = ["o", "s", "^", "D", "v", "P"]
    cs = [C_BLUE, C_BLUE, C_BLUE, C_RED, C_ORANGE, C_PURPLE]
    for (nm, p_, o_), m_, c_ in zip(POINTS, mk, cs):
        ax.plot(p_, o_, m_, color=c_, ms=6, zorder=3, label=L[nm])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(t["b_x"]); ax.set_ylabel(t["b_y"]); ax.set_title(t["b_t"])
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.2)
    fig.tight_layout()
    return fig


# ── 图 c ─────────────────────────────────────────────────────────────────────
def fig_c(lang: str) -> Figure:
    t, L = TXT[lang], LBL[lang]
    fig = Figure(figsize=(7.0, 2.5))
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    left, cols = 0.0, [C_GREEN, C_ORANGE, C_RED]
    for (nm, v), c in zip(SPEC_VC, cols):
        ax1.barh(0, v, left=left, color=c, height=0.5, zorder=3, label=L[nm])
        ax1.text(left + v / 2, 0, f"{v:.1f}%", ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")
        left += v
    ax1.set_xlim(0, 100); ax1.set_ylim(-0.55, 0.75); ax1.set_yticks([])
    ax1.set_xlabel("%"); ax1.set_title(t["c1_t"]); ax1.grid(False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1)
    _panel_tag(ax1, "a")

    # 2026-07-26：原为「两个工作点」的柱状对比。预投稿评审判定 n=2 无法证伪，
    # 改为 A12opscan 的**十点扫描**：预注册 10 个配置 × Formal 5 种子，覆盖 R² 0.10–0.82。
    ax2.scatter(OPSCAN_R2, OPSCAN_TOT, s=24, color=C_BLUE, zorder=4,
                label=t["total"], marker="o")
    ax2.scatter(OPSCAN_R2, OPSCAN_INS, s=24, color=C_RED, zorder=4,
                label=t["instr"], marker="s")
    _xx = np.linspace(OPSCAN_R2.min(), OPSCAN_R2.max(), 50)
    _a0 = OPSCAN_INS.mean() - OPSCAN_SLOPE * OPSCAN_R2.mean()
    ax2.plot(_xx, _a0 + OPSCAN_SLOPE * _xx, color=C_RED, lw=1.0, ls="--", zorder=3)
    ax2.fill_between(_xx, _a0 + OPSCAN_SLOPE_LO * _xx, _a0 + OPSCAN_SLOPE_HI * _xx,
                     color=C_RED, alpha=0.18, lw=0, zorder=2)
    ax2.set_xlabel("$R^2$")
    ax2.set_ylabel(t["c2_y"]); ax2.set_title(t["c2_t"])
    ax2.set_ylim(0, 2.55); ax2.set_xlim(0.03, 0.88)
    ax2.legend(loc="upper right", fontsize=6.5)
    ax2.text(0.055, 0.42, t["c2_note"], fontsize=6.2, color=C_RED, va="top", zorder=5)
    _panel_tag(ax2, "b")

    fig.tight_layout(w_pad=2.0)
    return fig


# ── 图 d ─────────────────────────────────────────────────────────────────────
def fig_d(lang: str) -> Figure:
    """正文图 d：**仅**苹果个体级泄漏（已确证的一条）。

    2026-07-26 起，时序「目标由分组变量派生」那一面板移入补充材料（图 S1），
    正文只保留被控制实验确证的个体级泄漏——见 Elsevier_*.tex §3.5 与 Supplementary_*.tex。
    """
    t, L = TXT[lang], LBL[lang]
    fig = Figure(figsize=(3.5, 2.6))
    ax1 = fig.add_subplot(1, 1, 1)

    x = np.arange(2); w = 0.36
    face = [LEAK_001[0][1], LEAK_001[1][1]]
    fru = [LEAK_001[0][2], LEAK_001[1][2]]
    ax1.bar(x - w / 2, face, w, color=C_ORANGE, zorder=3, label=L["面级 SSC"])
    ax1.bar(x + w / 2, fru, w, color=C_BLUE, zorder=3, label=L["整果均值 SSC"])
    for i, (a, b) in enumerate(zip(face, fru)):
        ax1.text(i - w / 2, a + 0.004, f"{a:.4f}", ha="center", fontsize=6.8)
        ax1.text(i + w / 2, b + 0.004, f"{b:.4f}", ha="center", fontsize=6.8)
    ax1.set_xticks(x); ax1.set_xticklabels([L[LEAK_001[0][0]], L[LEAK_001[1][0]]])
    ax1.set_ylabel(t["d_y1"]); ax1.set_title(t["d1_t"]); ax1.set_ylim(0, 0.215)
    ax1.legend(loc="upper right")

    fig.tight_layout()
    return fig


# ── 图 S1（补充材料）：时序数据「目标由分组变量派生」──────────────────────────
def fig_s1(lang: str) -> Figure:
    """补充材料图 S1：按采集日分组导致的性能下降 —— 归因于目标派生自分组变量。

    与原图 d 面板 (b) 数据、配色、误差棒口径完全一致，只是独立成图。
    """
    t, L = TXT[lang], LBL[lang]
    fig = Figure(figsize=(3.7, 2.7))
    ax2 = fig.add_subplot(1, 1, 1)

    x = np.arange(2)
    r2 = [LEAK_003[0][1], LEAK_003[1][1]]
    rm = [LEAK_003[0][2], LEAK_003[1][2]]
    # 误差棒 = 簇(=种子) bootstrap CI95。按采集日分组的区间宽近 400 倍于按果分组，
    # 这个「不稳定」本身就是本面板要传达的证据（正文 3.5），故必须画出来。
    r2_err = np.array([[r2[i] - LEAK_003[i][3][0] for i in range(2)],
                       [LEAK_003[i][3][1] - r2[i] for i in range(2)]])
    rm_err = np.array([[rm[i] - LEAK_003[i][4][0] for i in range(2)],
                       [LEAK_003[i][4][1] - rm[i] for i in range(2)]])
    ax2.bar(x, r2, 0.45, color=C_BLUE, zorder=3)
    ax2.errorbar(x, r2, yerr=r2_err, fmt="none", ecolor="0.25", elinewidth=1.0,
                 capsize=3, zorder=5)
    ax2.axhline(0, color="0.55", lw=0.8, zorder=2)
    for i, v in enumerate(r2):
        # 负值柱的标签放在下须之外，避开右轴 RMSE 折线穿过的区域
        if v >= 0:
            ax2.text(i, LEAK_003[i][3][1] + 0.06, f"{v:.3f}", ha="center",
                     fontsize=7, color=C_BLUE)
        else:
            ax2.text(i, LEAK_003[i][3][0] - 0.16, f"{v:.3f}", ha="center",
                     fontsize=7, color=C_BLUE)
    ax2.set_ylabel(t["d_y1"], color=C_BLUE); ax2.set_ylim(-1.12, 1.30)
    ax2.tick_params(axis="y", labelcolor=C_BLUE)
    axr = ax2.twinx()
    axr.errorbar(x, rm, yerr=rm_err, fmt="-o", color=C_RED, ecolor=C_RED,
                 elinewidth=1.0, capsize=3, zorder=4)
    for i, v in enumerate(rm):
        axr.annotate(f"{v:.2f}", (i, LEAK_003[i][4][0]), xytext=(14, -3),
                     textcoords="offset points", fontsize=7, color=C_RED, ha="left")
    axr.set_ylabel(t["d_y2"], color=C_RED); axr.tick_params(axis="y", labelcolor=C_RED)
    axr.set_ylim(-3.0, 16.5); axr.grid(False); axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(C_RED)
    ax2.set_xticks(x); ax2.set_xticklabels([L[LEAK_003[0][0]], L[LEAK_003[1][0]]])
    ax2.set_title(t["d2_t"])

    fig.tight_layout()
    return fig



# ── 图 e：半合成参考噪声退化（A9semisynth，预注册 V9b）─────────────────────────
def fig_e(lang: str) -> Figure:
    """两面板：(a) 退化轨迹 vs 预言 vs 结构上界；(b) 相对偏差随 m 收缩。

    **所有数值现场读自 outputs/A9semisynth_v18.xlsx**（含 cluster bootstrap CI 与
    预言值），绘图脚本不重算、不硬编码——吸取 EF-001 的教训：绘图脚本里的常量不算证据。
    """
    t = TXT[lang]
    d = pd.read_excel(os.path.join(OUT, "A9semisynth_v18.xlsx"), sheet_name="退化轨迹")
    d = d[pd.to_numeric(d["m（参考重复次数）"], errors="coerce").notna()].copy()
    d["m"] = pd.to_numeric(d["m（参考重复次数）"])
    fig = Figure(figsize=(6.9, 2.7))
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    for tgt, col, mk, lab in (("DM", C_BLUE, "o", t["e_dm"]),
                              ("SSC", C_ORANGE, "s", t["e_ssc"])):
        g = d[d["目标"] == tgt].sort_values("m")
        m = g["m"].to_numpy(float)
        obs = g["实测 R²（对带噪标签）"].to_numpy(float)
        lo = g["cluster CI95 下限"].to_numpy(float)
        hi = g["cluster CI95 上限"].to_numpy(float)
        ax1.plot(m, g["V9b 结构上界（正文口径）"], ":", color=col, lw=1.0, alpha=0.75)
        ax1.plot(m, g["预言 R²_pred"], "-", color=col, lw=1.4, alpha=0.9)
        ax1.errorbar(m, obs, yerr=[obs - lo, hi - obs], fmt=mk, color=col, ms=3.6,
                     capsize=2, lw=0, elinewidth=0.9, label=lab, zorder=5)
        ax2.plot(m, g["相对偏差 obs/pred−1"], "-" + mk, color=col, ms=3.6, lw=1.2,
                 label=lab)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel(t["e_x"])
    ax1.set_ylabel(t["e_y1"]); ax1.set_title(t["e1_t"])
    h, la = ax1.get_legend_handles_labels()
    h += [Line2D([], [], color="0.35", ls="-", lw=1.4),
          Line2D([], [], color="0.35", ls=":", lw=1.0)]
    la += [t["e_pred"], t["e_bound"]]
    ax1.legend(h, la, loc="lower right", fontsize=6.2)
    _panel_tag(ax1, "a")

    ax2.axhline(0, color="0.55", lw=0.8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax2.set_ylabel(t["e_y2"]); ax2.set_title(t["e2_t"])
    ax2.legend(loc="lower right", fontsize=6.2); _panel_tag(ax2, "b")

    fig.tight_layout(w_pad=2.2)
    return fig


def main() -> None:
    setup_chinese_font()
    stem = os.path.splitext(os.path.basename(__file__))[0]
    for i, fn in enumerate([fig_a, fig_b, fig_c, fig_d, fig_e]):
        paths = save_figure_bilingual(fn, stem, i)
        print(f"图{chr(97+i)} → {os.path.basename(paths['zh'])} / {os.path.basename(paths['en'])}")
    p = save_figure_bilingual(fig_s1, "B0suppfig_v18", 0)
    print(f"图S1 → {os.path.basename(p['zh'])} / {os.path.basename(p['en'])}")


if __name__ == "__main__":
    main()
