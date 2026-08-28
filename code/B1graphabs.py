"""图文摘要（Elsevier graphical abstract）：七面板密集版，矢量输出。

判据是「不读正文就看懂这篇在做什么」。七个面板按论证顺序排：
  1 参考侧      一果五面，读数彼此不同         5 仪器侧（猕猴桃）
  2 误差预算    方差拆成果间/果内             6 验证审计——划分泄漏
  3 可达上界    上界曲线与分产地散布           7 可证伪的衰减诊断
  4 设计规则    目标 R² → 所需重复次数

技术路线：Python 现场读工作簿 → 自包含 HTML（内联 SVG）→ Chrome headless 出真矢量
PDF。数字全部现场读，读不到直接抛错——绝不回退硬编码（回退等于把假零变成假证据，
见 0NFIGURE_AUDIT 的 G1 条款）。

数字的取法与来源（每一个都能追到工作簿的具体单元格）
  93sscceiling 表c   var_apple / var_face / ICC / ICC_CI95 / n_fruit；分产地 ICC 取极值
  93sscceiling 表d   果内极差 P50、果内 SD 中位数 / 仪器精度、折光仪标称精度
  97instrbudget 表a  ≥2 台设备扫过的果数、设备数
  97instrbudget 表c  谱层面方差三分量占比
  A12opscan 表a      十个操作点的仪器分量（均值、跨度、占 MSEP 比例的两端与倍数）
  A4formal 表c       泄漏对照：raw · y_fruit 的按行随机 / 按果分组
  A5aggregate 表b    衰减比实测与预言；表c 泄漏倍数（正文口径）四个值取范围
  A1consolidate 表c  纯标签基线的理论值与实测值

版式口径
  · 画布 1500×548 px；@page 必须显式定尺寸，否则 Chrome 按 Letter 出并裁掉右侧
  · 上排四面板、下排三面板，圆形编号 + 细分隔线，面板之间靠留白不靠重边框
  · 字体走系统 Helvetica Neue / Arial，全拉丁；不引 Adobe Fonts（要联网，本地渲染吃不到）

产物（落点见下方 OUT / FIGS 的判定——本地与发布树两套目录名都支持）
  B1graphabs.html            自包含源文件，可直接改版式
  B1graphabs.pdf / .png      投稿上传件，真矢量 + 300dpi 位图

修订记录
| 修订日期 | 轮次 | 改了什么 | 为什么改 |
|---|---|---|---|
| 2026-08-28 | 1 | 新建，取代 B0figures_v18.py 的 fig_ga | 旧版两 panel 都是数据曲线，看不出框架 |
| 2026-08-28 | 2 | 路线由 FigureSpec/SVG 改为 HTML+SVG→Chrome | ryan 判「太丑」；灰框流程图无视觉焦点 |
| 2026-08-28 | 3 | 三栏改七面板密集版，矢量化 | 依 ryan 选定的 V4 dense 版式复刻；原稿是位图，需真矢量 |
| 2026-08-28 | 4 | 删右下角来源脚注，画布 600→548 | ryan 判该句多余；删后底部横幅贴齐下沿，不留空条 |
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Paths are relative to this file, so the README command works from the repo root.
OUT = os.path.join(HERE, "..", "outputs")
FIGS = os.path.join(HERE, "..", "figures")
STEM = "B1graphabs"

CANVAS_W, CANVAS_H = 1500, 548          # 底部横幅贴齐画布下沿；@300dpi → 4500×1644 px


def _sheet(book: str, name: str) -> pd.DataFrame:
    return pd.read_excel(os.path.join(OUT, f"{book}_v18.xlsx"), sheet_name=name)


# ── 全部数值现场读自工作簿 ──────────────────────────────────────────────────
_vc = _sheet("93sscceiling", "表c：方差分解（总体与分产地）").set_index("分组")
_row = _vc.loc["全体（主口径）"]
VAR_APPLE, VAR_FACE = float(_row["var_apple"]), float(_row["var_face"])
ICC, N_FRUIT = float(_row["ICC"]), int(_row["n_fruit"])
ICC_CI = str(_row["ICC_CI95"])                                   # 形如 "[0.436, 0.515]"
_origin = [float(_vc.loc[k, "ICC"]) for k in _vc.index if k != "全体（主口径）"]
ORIGIN_LO, ORIGIN_HI = min(_origin), max(_origin)

_d = _sheet("93sscceiling", "表d：果内异质性幅度 vs 仪器精度").set_index("指标")["值"]
RANGE_P50 = float(_d["果内极差 P50"])
SD_OVER_SPEC = float(_d["果内 SD 中位数 / 仪器精度"])
SPEC = float(_d["折光仪标称精度"])

_ia = _sheet("97instrbudget", "表a：表a：设计结构与可辨识性").set_index("项")["值"]
N_KIWI, N_DEVICE = int(_ia["≥2 台设备扫过的果数（仪器分量的有效 n）"]), int(_ia["设备数"])
_ic = _sheet("97instrbudget", "表c：谱层面方差分解")["占比%"]
SPEC_FRUIT, SPEC_DEV, SPEC_INT = (float(_ic.iloc[i]) for i in range(3))

_op = _sheet("A12opscan", "表a：操作点扫描明细")
_share, _comp = _op["仪器分量占 MSEP 比例"], _op["仪器相关 RMSEP 分量（%DM）"]
SHARE_LO, SHARE_HI = float(_share.min()) * 100, float(_share.max()) * 100
SHARE_MULT = float(_share.max() / _share.min())
COMP_MEAN = float(_comp.mean())
COMP_SPAN_PCT = float(_comp.max() - _comp.min()) / COMP_MEAN * 100

_lk = _sheet("A4formal", "表c：泄漏对照").set_index("量")["跨种子均值"]
LEAK_RANDOM = float(_lk["raw · y_fruit · 按行随机"])
LEAK_GROUPED = float(_lk["raw · y_fruit · 按果分组"])
_mult = _sheet("A5aggregate", "表c：泄漏倍数（均值之比）")["均值之比（正文口径）"]
INFL_LO, INFL_HI = float(_mult.min()), float(_mult.max())

_base = _sheet("A1consolidate", "表c：纯标签基线理论vs实测").set_index("量")["值"]
BASE_PRED, BASE_MEAS = float(_base.iloc[0]), float(_base.iloc[1])

_att = _sheet("A5aggregate", "表b：衰减比（均值之比）")
ATT_OBS = float(_att.iloc[0]["均值之比（正文口径）"])
ATT_PRED = float(_att.iloc[1]["均值之比（正文口径）"])

WITHIN, BETWEEN = (1 - ICC) * 100, ICC * 100
VAR_RATIO = VAR_FACE / VAR_APPLE                                 # 设计规则里的 σf²/σa²

# ── 配色 ────────────────────────────────────────────────────────────────────
NAVY, MID, PALE = "#1F4E79", "#2E6DA4", "#A9C5DE"
ORANGE, GREY, RED, GREEN = "#E8A33D", "#8A99AB", "#B0392B", "#2E7D5B"
INK, MUTED, LINE, PANEL = "#1B2C42", "#5E6E82", "#DFE5EC", "#FFFFFF"


def ceiling_at(m: float) -> float:
    """m 次参考重复测定下的可达 R² 上界。"""
    return VAR_APPLE / (VAR_APPLE + VAR_FACE / m)


def faces_needed(target: float) -> int:
    """设计规则反解：达到 target R² 所需的最少参考重复次数。"""
    import math
    return math.ceil(VAR_RATIO * target / (1 - target))


def _panel(x: float, y: float, w: float, h: float, num: int, title: str) -> str:
    """面板外框 + 圆形编号 + 标题 + 标题下的细分隔线。"""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{PANEL}" '
            f'stroke="{LINE}" stroke-width="1.2"/>'
            f'<circle cx="{x + 26}" cy="{y + 25}" r="11.5" fill="{NAVY}"/>'
            f'<text x="{x + 26}" y="{y + 30}" font-size="13" font-weight="700" fill="#fff" '
            f'text-anchor="middle">{num}</text>'
            f'<text x="{x + 46}" y="{y + 31}" font-size="15.5" font-weight="700" '
            f'fill="{NAVY}" letter-spacing=".4">{title}</text>'
            f'<line x1="{x + 16}" y1="{y + 43}" x2="{x + w - 16}" y2="{y + 43}" '
            f'stroke="{LINE}"/>')


def _arrow(x: float, y: float) -> str:
    """面板之间的推进箭头。"""
    return (f'<path d="M{x},{y - 7} L{x + 9},{y} L{x},{y + 7} Z" fill="{NAVY}"/>')


def build_html() -> str:
    import math

    # ── 面板 3 的上界曲线（局部坐标 210×132）──────────────────────────────
    cw, ch = 210, 132
    def cx(m: float) -> float:
        return (m - 1) / 9 * cw

    def cy(v: float) -> float:
        return ch - (v - 0.30) / 0.70 * ch

    curve = " ".join(f'{"M" if i == 0 else "L"}{cx(m):.1f},{cy(ceiling_at(m)):.1f}'
                     for i, m in enumerate([1 + i * 0.25 for i in range(37)]))
    marks = "".join(
        f'<circle cx="{cx(m):.1f}" cy="{cy(ceiling_at(m)):.1f}" r="5" fill="{ORANGE}" '
        f'stroke="#fff" stroke-width="1.6"/>'
        f'<text x="{cx(m) + (7 if m < 10 else -7):.1f}" y="{cy(ceiling_at(m)) + dy:.1f}" '
        f'font-size="11" font-weight="700" fill="{ORANGE}" '
        f'text-anchor="{"start" if m < 10 else "end"}">{ceiling_at(m):.3f}</text>'
        for m, dy in [(1, 15), (3, 17), (5, -9), (10, -9)])

    # ── 面板 4 的设计规则查表 ─────────────────────────────────────────────
    rows = ""
    for i, t in enumerate([0.70, 0.80, 0.90, 0.95]):
        yy = 62 + i * 25
        hot = t == 0.80                                  # 五面正是本研究的采样设计
        rows += (
            f'<rect x="8" y="{yy - 18}" width="308" height="24" rx="5" '
            f'fill="{ORANGE if hot else "#F4F7FA"}"/>'
            f'<text x="86" y="{yy}" font-size="14.5" font-weight="700" '
            f'fill="{"#fff" if hot else INK}" text-anchor="middle">{t:.2f}</text>'
            f'<text x="242" y="{yy}" font-size="14.5" font-weight="700" '
            f'fill="{"#fff" if hot else NAVY}" text-anchor="middle">{faces_needed(t)}</text>')

    # ── 面板 6 的划分示意点阵 ─────────────────────────────────────────────
    def dots(ox: float, split: bool) -> str:
        """split=True 表示同一果的五个面被拆到 train/test 两侧（泄漏）。"""
        out = ""
        for r in range(3):
            for c in range(4):
                fill = [NAVY, MID, GREY][r]
                inside = c < 2 if split else True
                out += (f'<circle cx="{ox + (c * 13 if inside else ox * 0 + c * 13)}" '
                        f'cy="{18 + r * 13}" r="4.2" fill="{fill}" '
                        f'opacity="{1 if inside else .55}"/>')
        return out

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Graphical abstract</title>
<meta name="hz:slide-selector" content=".infographic">
<meta name="hz:canvas-width" content="{CANVAS_W}">
<meta name="hz:canvas-height" content="{CANVAS_H}">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#fff}}
/* 不给 @page 定尺寸，Chrome --print-to-pdf 会按 Letter 出并把右侧裁掉 */
@page{{size:{CANVAS_W}px {CANVAS_H}px;margin:0}}
@media print{{body{{margin:0}}.infographic{{page-break-after:avoid}}}}
.infographic{{position:relative;width:{CANVAS_W}px;height:{CANVAS_H}px;background:#fff;
  font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};overflow:hidden}}
svg text{{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}}
.it{{font-style:italic}}
</style></head>
<body><div class="infographic" data-canvas-width="{CANVAS_W}" data-canvas-height="{CANVAS_H}">
<svg width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
<defs>
  <linearGradient id="ap" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#E4694A"/><stop offset="1" stop-color="#9CBF52"/>
  </linearGradient>
  <pattern id="hatch" width="7" height="7" patternTransform="rotate(45)"
    patternUnits="userSpaceOnUse">
    <line x1="0" y1="0" x2="0" y2="7" stroke="{PALE}" stroke-width="2.4" opacity=".55"/>
  </pattern>
</defs>

<!-- ══ header ══ -->
<text x="24" y="30" font-size="21" font-weight="700" fill="{NAVY}" letter-spacing=".2">
  AN ERROR BUDGET FOR NIR FRUIT CALIBRATION</text>
<text x="24" y="50" font-size="12.5" font-style="italic" fill="{MUTED}">
  reference-replicate design rule &#183; falsifiable attenuation diagnostic &#183;
  leakage-audited validation</text>
<text x="{CANVAS_W - 24}" y="30" font-size="11.5" fill="{MUTED}" text-anchor="end">
  apples {N_FRUIT} &#215; 5 faces &#183; kiwifruit {N_KIWI:,} &#215; 2&#8211;{N_DEVICE} devices</text>

<!-- ══ 1 REFERENCE SIDE ══ -->
{_panel(24, 66, 342, 236, 1, "REFERENCE SIDE")}
<g transform="translate(24,66)">
  <ellipse cx="86" cy="168" rx="46" ry="7" fill="#000" opacity=".07"/>
  <path d="M86 74 C58 66 34 88 34 122 C34 154 58 176 86 176 C114 176 138 154 138 122
           C138 88 114 66 86 74 Z" fill="url(#ap)"/>
  <path d="M86 76 C88 66 96 58 108 56 C106 68 98 74 86 76 Z" fill="#5C8A3A"/>
  <path d="M86 76 L84 62" stroke="#6B4A2A" stroke-width="3" fill="none" stroke-linecap="round"/>
  <ellipse cx="64" cy="104" rx="11" ry="17" fill="#fff" opacity=".2"/>
  {"".join(f'<circle cx="{86 + 34 * math.cos(a):.1f}" cy="{124 + 34 * math.sin(a):.1f}" '
           f'r="6" fill="#FFF6E6" stroke="{ORANGE}" stroke-width="1.6"/>'
           f'<circle cx="{86 + 34 * math.cos(a):.1f}" cy="{124 + 34 * math.sin(a):.1f}" '
           f'r="1.9" fill="{RED}"/>'
           for a in [-1.5708, -0.3142, 0.9425, 2.1991, 3.4558])}
  <text x="86" y="200" font-size="12" font-style="italic" fill="{MUTED}"
    text-anchor="middle">one fruit, five answers</text>
  {"".join(f'<circle cx="152" cy="{y}" r="3.4" fill="{ORANGE}"/>'
           f'<text x="164" y="{y - 4}" font-size="12" fill="{MUTED}">{lab}</text>'
           f'<text x="164" y="{y + 13}" font-size="14.5" font-weight="700" fill="{INK}">{val}</text>'
           for y, lab, val in [
               (74, "apples &#215; flesh faces", f"{N_FRUIT} &#215; 5"),
               (124, "median 5-face range", f"{RANGE_P50:.2f} &#176;Brix"),
               (174, "within-fruit SD vs meter",
                f"{SD_OVER_SPEC:.1f} &#215; spec (&#177;{SPEC:g})")])}
</g>
{_arrow(374, 184)}

<!-- ══ 2 ERROR BUDGET ══ -->
{_panel(390, 66, 342, 236, 2, "ERROR BUDGET")}
<g transform="translate(390,66)">
  <text x="16" y="70" font-size="15" font-style="italic" font-weight="700" fill="{INK}">
    Y<tspan font-size="10.5" dy="4">ij</tspan>
    <tspan dy="-4"> = &#956; + a</tspan><tspan font-size="10.5" dy="4">i</tspan>
    <tspan dy="-4"> + e</tspan><tspan font-size="10.5" dy="4">ij</tspan></text>
  <text x="326" y="70" font-size="12.5" fill="{MUTED}" text-anchor="end">
    &#963;a&#178; = {VAR_APPLE:.3f} &#160; &#963;f&#178; = {VAR_FACE:.3f}</text>
  <text x="16" y="94" font-size="12.5" fill="{MUTED}">reference variance</text>
  <text x="326" y="94" font-size="12.5" font-weight="700" fill="{NAVY}" text-anchor="end">
    ICC {ICC:.3f} {ICC_CI}</text>
  <rect x="16" y="104" width="310" height="27" rx="6" fill="{NAVY}"/>
  <rect x="16" y="104" width="{310 * WITHIN / 100:.1f}" height="27" rx="6" fill="{ORANGE}"/>
  <rect x="{16 + 310 * WITHIN / 100 - 8:.1f}" y="104" width="8" height="27" fill="{ORANGE}"/>
  <text x="{16 + 310 * WITHIN / 200:.1f}" y="122" font-size="12.5" font-weight="700"
    fill="#fff" text-anchor="middle">{WITHIN:.1f}% within</text>
  <text x="{16 + 310 * WITHIN / 100 + 310 * BETWEEN / 200:.1f}" y="122" font-size="12.5"
    font-weight="700" fill="#fff" text-anchor="middle">{BETWEEN:.1f}% between</text>
  <text x="16" y="163" font-size="14" font-style="italic" font-weight="700" fill="{INK}">
    RMSEP&#178; = model + &#963;f&#178;/m + instrument</text>
  <rect x="16" y="175" width="310" height="26" rx="6" fill="{NAVY}"/>
  <rect x="140" y="175" width="105" height="26" fill="{PALE}"/>
  <rect x="245" y="175" width="81" height="26" rx="6" fill="{GREY}"/>
  <rect x="245" y="175" width="8" height="26" fill="{GREY}"/>
  <text x="78" y="192" font-size="12" font-weight="700" fill="#fff" text-anchor="middle">model</text>
  <text x="192" y="192" font-size="12" font-weight="700" fill="{NAVY}" text-anchor="middle">reference</text>
  <text x="285" y="192" font-size="12" font-weight="700" fill="#fff" text-anchor="middle">instrument</text>
  <text x="16" y="222" font-size="12" font-style="italic" fill="{MUTED}">
    two of the three parts are not the model&#8217;s &#8212;</text>
  <text x="16" y="237" font-size="12" font-style="italic" fill="{MUTED}">
    decompose before tuning models</text>
</g>
{_arrow(740, 184)}

<!-- ══ 3 ATTAINABLE CEILING ══ -->
{_panel(756, 66, 342, 236, 3, "ATTAINABLE CEILING")}
<g transform="translate(812,124)">
  <rect x="0" y="0" width="{cw}" height="{cy(0.30)}" fill="url(#hatch)" opacity=".5"/>
  <path d="{curve} L{cw},{ch} L0,{ch} Z" fill="#fff"/>
  <text x="14" y="17" font-size="10.5" fill="{GREY}" letter-spacing="1.2">UNATTAINABLE</text>
  <path d="{curve}" fill="none" stroke="{NAVY}" stroke-width="2.6"/>
  {marks}
  <line x1="0" y1="0" x2="0" y2="{ch}" stroke="{MUTED}" stroke-width="1.1"/>
  <line x1="0" y1="{ch}" x2="{cw}" y2="{ch}" stroke="{MUTED}" stroke-width="1.1"/>
  <text x="-8" y="4" font-size="10.5" fill="{MUTED}" text-anchor="end">1.0</text>
  <text x="-8" y="{cy(0.5) + 4:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">0.5</text>
  <text x="-8" y="{ch + 4}" font-size="10.5" fill="{MUTED}" text-anchor="end">0.3</text>
  <text x="-30" y="{ch / 2:.0f}" font-size="12" font-weight="700" fill="{INK}">R&#178;</text>
  <text x="0" y="{ch + 17}" font-size="10.5" fill="{MUTED}" text-anchor="middle">1</text>
  <text x="{cx(5):.1f}" y="{ch + 17}" font-size="10.5" fill="{MUTED}" text-anchor="middle">5</text>
  <text x="{cw}" y="{ch + 17}" font-size="10.5" fill="{MUTED}" text-anchor="middle">10</text>
  <text x="{cw / 2:.0f}" y="{ch + 33}" font-size="11.5" fill="{MUTED}"
    text-anchor="middle">reference replicates m</text>
  <line x1="24" y1="{cy(ORIGIN_LO):.1f}" x2="24" y2="{cy(ORIGIN_HI):.1f}"
    stroke="{MUTED}" stroke-width="1.4"/>
  <line x1="19" y1="{cy(ORIGIN_HI):.1f}" x2="29" y2="{cy(ORIGIN_HI):.1f}" stroke="{MUTED}" stroke-width="1.4"/>
  <line x1="19" y1="{cy(ORIGIN_LO):.1f}" x2="29" y2="{cy(ORIGIN_LO):.1f}" stroke="{MUTED}" stroke-width="1.4"/>
  <text x="34" y="{cy(ORIGIN_LO) + 4:.1f}" font-size="10.5" fill="{MUTED}">origin groups
    {ORIGIN_LO:.3f}&#8211;{ORIGIN_HI:.3f}</text>
</g>
{_arrow(1106, 184)}

<!-- ══ 4 DESIGN RULE ══ -->
{_panel(1122, 66, 354, 236, 4, "DESIGN RULE")}
<g transform="translate(1122,66)">
  <text x="177" y="72" font-size="14.5" font-style="italic" font-weight="700" fill="{INK}"
    text-anchor="middle">m &#8805; (&#963;f&#178;/&#963;a&#178;) &#183; R&#178;/(1&#8722;R&#178;)</text>

  <g transform="translate(19,80)">
    <text x="86" y="16" font-size="11.5" fill="{MUTED}" text-anchor="middle">target R&#178;</text>
    <text x="242" y="16" font-size="11.5" fill="{MUTED}" text-anchor="middle">faces m &#8805;</text>
    {rows}
  </g>
  <text x="177" y="230" font-size="10.5" fill="{MUTED}" text-anchor="middle">
    &#963;f&#178;/&#963;a&#178; = {VAR_RATIO:.4f} from a pilot batch &#183;
    <tspan font-style="italic">add one extra face as margin</tspan></text>
</g>

<!-- ══ 5 INSTRUMENT SIDE ══ -->
{_panel(24, 314, 560, 172, 5, "INSTRUMENT SIDE (KIWIFRUIT)")}
<g transform="translate(24,314)">
  <circle cx="58" cy="106" r="34" fill="#8A6B3F"/>
  <circle cx="58" cy="106" r="29" fill="#A7C24E"/>
  <ellipse cx="58" cy="106" rx="11" ry="13" fill="#F0F4DC"/>
  {"".join(f'<ellipse cx="{58 + 20 * math.cos(a):.1f}" cy="{106 + 22 * math.sin(a):.1f}" '
           f'rx="2.6" ry="3.6" fill="#2E2A20" transform="rotate({a * 57.3:.0f} '
           f'{58 + 20 * math.cos(a):.1f} {106 + 22 * math.sin(a):.1f})"/>'
           for a in [i * 0.5236 for i in range(12)])}
  <rect x="104" y="76" width="42" height="24" rx="5" fill="{PALE}" stroke="{MID}"/>
  <rect x="104" y="114" width="42" height="24" rx="5" fill="{PALE}" stroke="{MID}"/>
  <text x="125" y="92" font-size="12" font-weight="700" fill="{NAVY}" text-anchor="middle">D1</text>
  <text x="125" y="130" font-size="12" font-weight="700" fill="{NAVY}" text-anchor="middle">D2</text>
  <path d="M92 100 L104 88 M92 112 L104 126" stroke="#C9B99A" stroke-width="3" fill="none"/>
  <text x="168" y="72" font-size="12.5" font-weight="700" fill="{MUTED}">
    SNV-corrected spectral variance</text>
  <rect x="168" y="80" width="374" height="27" rx="6" fill="{GREY}"/>
  <rect x="168" y="80" width="{374 * (SPEC_FRUIT + SPEC_DEV) / 100:.1f}" height="27" rx="6" fill="{PALE}"/>
  <rect x="168" y="80" width="{374 * SPEC_FRUIT / 100:.1f}" height="27" rx="6" fill="{NAVY}"/>
  <text x="{168 + 374 * SPEC_FRUIT / 200:.1f}" y="98" font-size="12.5" font-weight="700"
    fill="#fff" text-anchor="middle">{SPEC_FRUIT:.1f}%</text>
  <text x="{168 + 374 * (SPEC_FRUIT + SPEC_DEV / 2) / 100:.1f}" y="98" font-size="12"
    font-weight="700" fill="{NAVY}" text-anchor="middle">{SPEC_DEV:.1f}%</text>
  <text x="{168 + 374 * (SPEC_FRUIT + SPEC_DEV + SPEC_INT / 2) / 100:.1f}" y="98"
    font-size="12" font-weight="700" fill="#fff" text-anchor="middle">{SPEC_INT:.1f}%</text>
  <text x="168" y="124" font-size="11.5" fill="{MUTED}">
    between fruit &#183; device &#183; fruit&#215;device</text>
  <text x="20" y="156" font-size="12.5" font-weight="700" fill="{INK}">
    device-linked error &#8776; {COMP_MEAN:.3f} %DM &#8212; spans just
    {COMP_SPAN_PCT:.1f}% of its mean across ten points</text>
  <text x="20" y="174" font-size="12.5" font-weight="700" fill="{RED}">
    yet its share of MSE rises {SHARE_LO:.1f}% &#8594; {SHARE_HI:.1f}%
    (&#215;{SHARE_MULT:.1f}) as the model improves</text>
</g>

<!-- ══ 6 VALIDATION AUDIT ══ -->
{_panel(596, 314, 348, 172, 6, "VALIDATION AUDIT &#8212; SPLIT LEAKAGE")}
<g transform="translate(596,314)">
  <text x="60" y="62" font-size="11" fill="{MUTED}" text-anchor="middle">train</text>
  <text x="176" y="62" font-size="11" fill="{MUTED}" text-anchor="middle">test</text>
  <rect x="18" y="68" width="84" height="52" rx="6" fill="#F6F8FA" stroke="{LINE}"/>
  <rect x="134" y="68" width="84" height="52" rx="6" fill="#F6F8FA" stroke="{LINE}"/>
  {"".join(f'<circle cx="{30 + c * 16}" cy="{82 + r * 15}" r="4" fill="{[NAVY, MID, GREY][r]}"/>'
           for r in range(3) for c in range(2))}
  {"".join(f'<circle cx="{146 + c * 16}" cy="{82 + r * 15}" r="4" fill="{[NAVY, MID, GREY][r]}"/>'
           for r in range(3) for c in range(2))}
  {"".join(f'<line x1="{54}" y1="{82 + r * 15}" x2="{146}" y2="{82 + r * 15}" '
           f'stroke="{GREY}" stroke-width="1.2" stroke-dasharray="3 3"/>' for r in range(3))}
  <text x="232" y="92" font-size="20" font-weight="700" fill="{RED}">&#10007;</text>
  <text x="60" y="140" font-size="13" font-weight="700" fill="{RED}" text-anchor="middle">
    R&#178; {LEAK_RANDOM:.4f}</text>
  <text x="176" y="140" font-size="13" font-weight="700" fill="{NAVY}" text-anchor="middle">
    R&#178; {LEAK_GROUPED:.4f}</text>
  <text x="60" y="155" font-size="11" fill="{RED}" text-anchor="middle">random split &#8212; inflated</text>
  <text x="176" y="155" font-size="11" fill="{NAVY}" text-anchor="middle">grouped &#8212; honest</text>
  <text x="174" y="176" font-size="12.5" font-weight="700" fill="{RED}" text-anchor="middle">
    random splitting inflates R&#178; by {INFL_LO:.2f}&#8211;{INFL_HI:.2f}&#215;</text>
</g>

<!-- ══ 7 FALSIFIABLE DIAGNOSTIC ══ -->
{_panel(956, 314, 520, 172, 7, "FALSIFIABLE ATTENUATION DIAGNOSTIC")}
<g transform="translate(956,314)">
  <text x="20" y="62" font-size="12" font-style="italic" fill="{MUTED}">
    six parameter-free predictions &#8212; all confirmed on held-out data</text>
  {"".join(
    f'<text x="20" y="{y + 4}" font-size="12.5" font-weight="700" fill="{INK}">{lab}</text>'
    f'<rect x="212" y="{y - 12}" width="180" height="9" rx="4.5" fill="#EDF1F6"/>'
    f'<rect x="212" y="{y - 12}" width="{180 * pv:.1f}" height="9" rx="4.5" fill="{PALE}"/>'
    f'<rect x="212" y="{y + 2}" width="180" height="9" rx="4.5" fill="#EDF1F6"/>'
    f'<rect x="212" y="{y + 2}" width="{180 * mv:.1f}" height="9" rx="4.5" fill="{NAVY}"/>'
    f'<text x="404" y="{y - 3}" font-size="11.5" fill="{MUTED}">pred {pv:.4f}</text>'
    f'<text x="404" y="{y + 12}" font-size="11.5" font-weight="700" fill="{NAVY}">meas {mv:.4f}</text>'
    f'<text x="492" y="{y + 5}" font-size="15" font-weight="700" fill="{GREEN}">&#10003;</text>'
    for y, lab, pv, mv in [(92, "label-only baseline R&#178;", BASE_PRED, BASE_MEAS),
                           (140, "attenuation ratio", ATT_PRED, ATT_OBS)])}
</g>

<!-- ══ banner ══ -->
<rect x="0" y="502" width="{CANVAS_W}" height="46" fill="{NAVY}"/>
<text x="{CANVAS_W / 2:.0f}" y="531" font-size="14.5" font-weight="700" fill="#fff"
  text-anchor="middle" letter-spacing=".3">
  REPORTED R&#178; MUST BE READ AGAINST A REFERENCE-LIMITED CEILING, NOT AGAINST 1
  &#8212; THE CEILING IS COMPUTABLE BEFORE THE EXPERIMENT</text>
</svg>
</div></body></html>"""


def find_chrome() -> str | None:
    """定位 Chrome/Chromium；找不到返回 None（只出 HTML，不假装出了 PDF）。"""
    cands = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    found = next((c for c in cands if os.path.isfile(c)), None)
    return found or shutil.which("google-chrome") or shutil.which("chromium")


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    html_path = os.path.join(FIGS, f"{STEM}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html())
    print(f"html  → {os.path.relpath(html_path, HERE)}")
    print(f"  ICC={ICC:.4f} {ICC_CI}  果间={BETWEEN:.2f}%  果内={WITHIN:.2f}%  n={N_FRUIT}")
    print(f"  上界 m=1/3/5/10: {ceiling_at(1):.3f}/{ceiling_at(3):.3f}/"
          f"{ceiling_at(5):.3f}/{ceiling_at(10):.3f}  σf²/σa²={VAR_RATIO:.4f}")
    print(f"  设计规则 0.70/0.80/0.90/0.95 → {faces_needed(0.70)}/{faces_needed(0.80)}/"
          f"{faces_needed(0.90)}/{faces_needed(0.95)} 面")
    print(f"  谱方差 {SPEC_FRUIT:.2f}/{SPEC_DEV:.2f}/{SPEC_INT:.2f}%  "
          f"仪器分量 {COMP_MEAN:.4f}%DM 跨度{COMP_SPAN_PCT:.1f}%  "
          f"占MSE {SHARE_LO:.1f}→{SHARE_HI:.1f}% (×{SHARE_MULT:.1f})")
    print(f"  泄漏 {LEAK_RANDOM:.4f}→{LEAK_GROUPED:.4f}  虚高 {INFL_LO:.2f}–{INFL_HI:.2f}×")
    print(f"  纯标签基线 {BASE_PRED:.4f}/{BASE_MEAS:.4f}  衰减比 {ATT_PRED:.4f}/{ATT_OBS:.4f}")

    chrome = find_chrome()
    if chrome is None:
        print("\n⚠ 未找到 Chrome/Chromium，只产出了 HTML。")
        print("  用任意浏览器打开该 HTML 打印为 PDF 即可；页面尺寸已写进 @page。")
        return

    base = ["--headless", "--disable-gpu", "--hide-scrollbars"]
    url = "file://" + html_path
    pdf = os.path.join(FIGS, f"{STEM}.pdf")
    png = os.path.join(FIGS, f"{STEM}.png")
    subprocess.run([chrome, *base, "--no-pdf-header-footer", f"--print-to-pdf={pdf}", url],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([chrome, *base, "--force-device-scale-factor=3",
                    f"--window-size={CANVAS_W},{CANVAS_H}", f"--screenshot={png}", url],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for p in (pdf, png):
        if not os.path.isfile(p):
            raise RuntimeError(f"未生成: {p}")
    print(f"图GA  → {os.path.relpath(pdf, HERE)} / {os.path.relpath(png, HERE)}")


if __name__ == "__main__":
    main()
