"""图形摘要（Elsevier graphical abstract）：论证骨架的三栏框架图。

与 B0figures_v18.py 的分工
--------------------------
B0 画的是**结果图**（曲线、误差棒、方差分解），读者要先读正文才知道那些曲线在说什么。
图形摘要的判据不同——它要让人**不读正文就看懂这篇在做什么**，所以本文件画的是
论证骨架：A 被忽略的前提 → B 把它拆成误差预算 → C 由此得到的两个可用工具。

技术路线：FigureSpec（JSON）→ SVG。JSON 是确定性中间产物，同一份 spec 永远渲染出同一
张图；数字全部现场读自工作簿，读不到直接抛错——绝不回退硬编码（回退等于把假零变成假
证据，见 0NFIGURE_AUDIT 的 G1 条款）。JSON 随产物一起留档，即便渲染器不在手边，图上每
个数字仍可逐个核回工作簿。

布局口径（figure-spec 的 anti-pattern 反着来）
  · 三栏顶边对齐，栏间隙 ≥16 px；
  · 栏间连线只走水平或短弧，不画跨行长对角；
  · 红色警示语做成 A 栏内的实体条——做成浮动标签会落在分组框外面；
  · 全图不用 ①②③：Helvetica 缺这些字，转 PDF 时会回退到中文字体（苹方），
    英文期刊图里不该嵌中文字体。分栏标签因此用 A./B./C.。

产物（落点见下方 OUT / FIGS 的判定——本地与发布树两套目录名都支持）
  B1graphabs.json            FigureSpec，数字可核
  B1graphabs.svg / .pdf      图本身，矢量

修订记录
| 修订日期 | 轮次 | 改了什么 | 为什么改 |
|---|---|---|---|
| 2026-08-28 | 1 | 新建，取代 B0figures_v18.py 的 fig_ga | 旧版两 panel 都是数据曲线，看不出框架 |
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd

Spec = dict[str, Any]

HERE = os.path.dirname(os.path.abspath(__file__))
# 发布树里 04outputs/ 叫 outputs/、01manuscript/figs/ 叫 figures/。两套名字都要能跑——
# README 的复现命令是在发布树里执行的，B0figures 之前正是在这里崩过。
_OUT_PUB = os.path.join(HERE, "..", "outputs")
OUT = _OUT_PUB if os.path.isdir(_OUT_PUB) else os.path.join(HERE, "..", "04outputs")
_FIG_PUB = os.path.join(HERE, "..", "figures")
FIGS = _FIG_PUB if os.path.isdir(_FIG_PUB) else os.path.join(
    HERE, "..", "06doc", "01manuscript", "figs")
STEM = "B1graphabs"


def _sheet(book: str, name: str) -> pd.DataFrame:
    return pd.read_excel(os.path.join(OUT, f"{book}_v18.xlsx"), sheet_name=name)


# ── 全部数值现场读自工作簿 ──────────────────────────────────────────────────
_vc = _sheet("93sscceiling", "表c：方差分解（总体与分产地）").set_index("分组")
_row = _vc.loc["全体（主口径）"]
VAR_APPLE, VAR_FACE = float(_row["var_apple"]), float(_row["var_face"])
ICC, N_FRUIT = float(_row["ICC"]), int(_row["n_fruit"])

_att = _sheet("A5aggregate", "表b：衰减比（均值之比）")
ATT_OBS = float(_att.iloc[0]["均值之比（正文口径）"])
ATT_LO = float(_att.iloc[0]["cluster CI95 下限"])
ATT_HI = float(_att.iloc[0]["cluster CI95 上限"])
ATT_PRED = float(_att.iloc[1]["均值之比（正文口径）"])

BETWEEN = ICC * 100.0                    # 果间方差占比 = 单面参考值的可达上界
WITHIN = (1.0 - ICC) * 100.0             # 果内方差占比 = 参考值自身的噪声
CEIL_1 = ICC                             # 单面参考值的可达上界
CEIL_5 = VAR_APPLE / (VAR_APPLE + VAR_FACE / 5.0)   # 五面均值作参考时的上界

# ── 配色：与 B0figures_v18.py 同族，但降饱和以适配大面积填充 ─────────────────
BLUE, ORANGE, GREEN, RED, GREY = "#3B6EA5", "#E08A2E", "#3F8F5C", "#C0392B", "#7A7A7A"
INK = "#222222"

TOP = 88            # 三栏内容的统一上边缘
ROW2 = TOP + 25     # B、C 两栏首行的中心纵坐标


def build_spec() -> Spec:
    return {
        "title": "Graphical abstract — an error budget for NIR fruit calibration",
        "canvas": {"width": 1130, "height": 476},
        "style": {
            "font_family": "Helvetica, Arial, sans-serif",
            "font_size": 13,
            "bg_color": "#FFFFFF",
        },
        "nodes": [
            # A. 被忽略的前提：两条通路汇到同一个 R² 比较
            {"id": "spec_in", "label": "NIR spectrum", "x": 96, "y": TOP + 21,
             "width": 128, "height": 42, "fill": "#EAF1F8", "stroke": BLUE,
             "text_color": INK, "font_size": 12.5},
            {"id": "model", "label": "calibration\nmodel", "sublabel": "prediction ŷ",
             "x": 238, "y": TOP + 21, "width": 116, "height": 56, "fill": "#EAF1F8",
             "stroke": BLUE, "text_color": INK, "font_size": 12.5},
            {"id": "fruit", "label": "one fruit", "x": 96, "y": TOP + 138,
             "width": 128, "height": 42, "fill": "#FBF0E2", "stroke": ORANGE,
             "text_color": INK, "font_size": 12.5},
            {"id": "faces", "label": "5 destructive\ndeterminations",
             "sublabel": "reference mean ȳ", "x": 240, "y": TOP + 138, "width": 132,
             "height": 56, "fill": "#FBF0E2", "stroke": ORANGE, "text_color": INK,
             "font_size": 12},
            {"id": "compare", "label": "R²  /  RMSEP",
             "sublabel": "judged against the reference",
             "x": 428, "y": TOP + 80, "width": 186, "height": 58,
             "fill": "#FFFFFF", "stroke": GREY, "text_color": INK, "font_size": 15},
            # 警示语做成节点而非浮动标签，才会被 A 栏的分组框包住
            {"id": "warn",
             "label": "the comparison treats the reference as exact — it is not",
             "x": 277, "y": TOP + 214, "width": 414, "height": 34,
             "fill": "#FBEAE8", "stroke": RED, "text_color": RED, "font_size": 13},

            # B. 误差预算：参考值的方差换算成天花板
            {"id": "between", "label": f"between-fruit\n{BETWEEN:.2f}%",
             "sublabel": "the signal — hence the ceiling", "x": 697, "y": ROW2,
             "width": 168, "height": 50,
             "fill": "#E7F2EA", "stroke": GREEN, "text_color": INK, "font_size": 12.5},
            {"id": "within", "label": f"within-fruit\n{WITHIN:.2f}%",
             "sublabel": "the reference’s own noise", "x": 697, "y": ROW2 + 84,
             "width": 138, "height": 50, "fill": "#FBF0E2", "stroke": ORANGE,
             "text_color": INK, "font_size": 12.5},
            {"id": "ceiling", "label": f"ceiling R² = {CEIL_1:.3f}",
             "sublabel": f"one face;  {CEIL_5:.3f} for the mean of all five",
             "x": 697, "y": ROW2 + 176, "width": 216, "height": 52,
             "fill": "#FBEAE8", "stroke": RED, "text_color": RED, "font_size": 14},

            # C. 两个交付物
            {"id": "tool1", "label": "Design rule",
             "sublabel": "target R² → replicates needed", "x": 983, "y": ROW2 + 2,
             "width": 220, "height": 54, "fill": "#EAF1F8", "stroke": BLUE,
             "text_color": INK, "font_size": 13.5},
            {"id": "tool2", "label": "Attenuation diagnostic",
             "sublabel": "no free parameter — falsifiable", "x": 983, "y": ROW2 + 176,
             "width": 220, "height": 54, "fill": "#EAF1F8", "stroke": BLUE,
             "text_color": INK, "font_size": 13.5},
            {"id": "tested",
             "label": f"measured {ATT_OBS:.3f} [{ATT_LO:.3f}, {ATT_HI:.3f}]",
             "sublabel": f"contains the predicted {ATT_PRED:.4f}; "
                         f"excludes {CEIL_1:.3f}",
             "x": 983, "y": ROW2 + 258, "width": 220, "height": 50, "fill": "#E7F2EA",
             "stroke": GREEN, "text_color": INK, "font_size": 11.5},
        ],
        "edges": [
            {"from": "spec_in", "to": "model", "color": BLUE, "thickness": 2},
            {"from": "model", "to": "compare", "color": BLUE, "thickness": 2},
            {"from": "fruit", "to": "faces", "color": ORANGE, "thickness": 2},
            {"from": "faces", "to": "compare", "color": ORANGE, "thickness": 2},
            {"from": "compare", "to": "between", "label": "decompose", "color": GREY,
             "thickness": 2},
            {"from": "compare", "to": "within", "color": GREY, "thickness": 2},
            {"from": "within", "to": "ceiling", "label": "caps R²", "color": RED,
             "thickness": 2},
            {"from": "ceiling", "to": "tool1", "label": "invert", "color": GREY,
             "thickness": 2, "curve": True},
            {"from": "ceiling", "to": "tool2", "label": "test", "color": GREY,
             "thickness": 2},
            {"from": "tool2", "to": "tested", "color": GREEN, "thickness": 2},
        ],
        "groups": [
            {"id": "g1", "label": "A.  The premise that is never checked",
             "node_ids": ["spec_in", "model", "fruit", "faces", "compare", "warn"],
             "fill": "#FAFBFC", "stroke": "#D6DCE2", "padding": 26},
            {"id": "g2", "label": "B.  Decompose it into an error budget",
             "node_ids": ["between", "within", "ceiling"],
             "fill": "#FAFBFC", "stroke": "#D6DCE2", "padding": 26},
            {"id": "g3", "label": "C.  Two tools, runnable on your own data",
             "node_ids": ["tool1", "tool2", "tested"],
             "fill": "#FAFBFC", "stroke": "#D6DCE2", "padding": 26},
        ],
        "labels": [
            {"text": f"{N_FRUIT} apples × 5 faces", "x": 240, "y": TOP + 176,
             "font_size": 11, "color": GREY, "anchor": "middle"},
            {"text": "Report calibration performance against a reference-limited "
                     "ceiling, not against unity",
             "x": 565, "y": 456, "font_size": 14.5, "color": INK, "anchor": "middle"},
        ],
    }


def check_layout(spec: Spec) -> None:
    """落点自检：节点两两不得重叠、分组不得越界、栏间隙不得过窄。

    渲染器的 validate 只查 schema，不查这三件事——初版正是栽在 model/faces 的右边缘
    越过了 compare 的左边缘上：图看着"只是箭头有点怪"，实际是两个节点叠在了一起。
    """
    nodes = {n["id"]: n for n in spec["nodes"]}

    def box(n: Spec) -> tuple[float, float, float, float]:
        return (n["x"] - n["width"] / 2, n["y"] - n["height"] / 2,
                n["x"] + n["width"] / 2, n["y"] + n["height"] / 2)

    ids = list(nodes)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = box(nodes[ids[i]]), box(nodes[ids[j]])
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                raise AssertionError(f"节点重叠: {ids[i]} × {ids[j]}")

    canvas_w, canvas_h = spec["canvas"]["width"], spec["canvas"]["height"]
    prev_right = None
    for g in spec["groups"]:
        bs = [box(nodes[i]) for i in g["node_ids"]]
        pad = g["padding"]
        x0, y0 = min(b[0] for b in bs) - pad, min(b[1] for b in bs) - pad
        x1, y1 = max(b[2] for b in bs) + pad, max(b[3] for b in bs) + pad
        if x0 < 0 or x1 > canvas_w or y0 < 0 or y1 > canvas_h:
            raise AssertionError(f"分组 {g['id']} 越出画布: x {x0:.0f}–{x1:.0f} "
                                 f"y {y0:.0f}–{y1:.0f}，画布 {canvas_w}×{canvas_h}")
        if prev_right is not None and x0 - prev_right < 12:
            raise AssertionError(f"分组 {g['id']} 与前一栏间隙仅 {x0 - prev_right:.0f}px")
        prev_right = x1


def find_renderer() -> str | None:
    """定位 figure-spec 的渲染器；找不到返回 None（只出 JSON，不假装出了图）。"""
    cands = []
    if os.environ.get("CLAUDE_SKILL_DIR"):
        cands.append(os.path.join(os.environ["CLAUDE_SKILL_DIR"], "scripts",
                                  "figure_renderer.py"))
    cands += [
        os.path.expanduser("~/.claude/skills/figure-spec/scripts/figure_renderer.py"),
        os.path.join(HERE, "..", ".aris", "tools", "figure_renderer.py"),
    ]
    return next((c for c in cands if os.path.isfile(c)), None)


def main() -> None:
    spec = build_spec()
    check_layout(spec)

    os.makedirs(FIGS, exist_ok=True)
    js = os.path.join(OUT, f"{STEM}.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    print(f"spec  → {os.path.relpath(js, HERE)}  "
          f"({len(spec['nodes'])} nodes / {len(spec['edges'])} edges)")
    print(f"  ICC={ICC:.4f}  果间={BETWEEN:.2f}%  果内={WITHIN:.2f}%  "
          f"上界 单面={CEIL_1:.3f} 五面均值={CEIL_5:.3f}  n={N_FRUIT}")
    print(f"  衰减比实测 {ATT_OBS:.4f} [{ATT_LO:.4f}, {ATT_HI:.4f}]  预言 {ATT_PRED:.4f}")

    renderer = find_renderer()
    if renderer is None:
        print("\n⚠ 未找到 figure_renderer.py（figure-spec skill），只产出了 JSON。")
        print("  安装该 skill 后重跑本脚本即可得到 SVG/PDF；"
              "JSON 里的数字已可逐个核回工作簿。")
        return

    svg = os.path.join(FIGS, f"{STEM}.svg")
    subprocess.run([sys.executable, renderer, "render", js, "--output", svg], check=True)

    # SVG → PDF：本机无 rsvg-convert / inkscape / cairosvg，走 LibreOffice
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        print("⚠ 未找到 soffice/libreoffice，跳过 PDF；SVG 已产出。")
        return
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", FIGS, svg],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = os.path.join(FIGS, f"{STEM}.pdf")
    if not os.path.isfile(pdf):
        raise RuntimeError(f"PDF 未生成: {pdf}")
    print(f"图GA  → {os.path.relpath(svg, HERE)} / {os.path.relpath(pdf, HERE)}")


if __name__ == "__main__":
    main()
