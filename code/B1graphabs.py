"""图文摘要（Elsevier graphical abstract）：三栏叙事 + 内联 SVG 图元。

判据是「不读正文就看懂这篇在做什么」，所以画的是论证骨架而非结果曲线的堆砌：
  A 从没被检查的前提 —— 参考值自己带噪声，而 R²/RMSEP 把它当准确值
  B 误差预算 —— 那点噪声把可达 R² 顶死在一个数上
  C 交付物 —— 一条设计规则、一个无自由参数因而可证伪的诊断

技术路线：Python 现场读工作簿 → 生成自包含 HTML（内联 SVG）→ Chrome headless 出
PDF/PNG。选 HTML 而非 matplotlib，是因为图文摘要要的是版式与字阶层次而不是坐标系；
选内联 SVG 而非位图，是为了矢量可缩放。数字全部现场读，读不到直接抛错——绝不回退
硬编码（回退等于把假零变成假证据，见 0NFIGURE_AUDIT 的 G1 条款）。

版式口径
  · 画布 1400×560 px（2.5:1）；@page 必须显式定尺寸，否则 Chrome 按 Letter 出并裁掉右栏
  · 三栏等宽，栏间只用一条 1px 竖线分隔——不用分组框：旧版 12 个灰底圆角框把画面切碎了
  · 核心数字（单面上界）做成 54px 视觉焦点，其余靠字重与色彩分层，不靠字号堆叠
  · 字体走系统 Avenir Next / Helvetica Neue，全拉丁；不引 Adobe Fonts（要联网，本地
    渲染吃不到），也不用 ①②③（Helvetica 缺字，转 PDF 会回退嵌入中文字体）

产物（落点见下方 OUT / FIGS 的判定——本地与发布树两套目录名都支持）
  B1graphabs.html            自包含源文件，可直接改版式
  B1graphabs.pdf / .png      投稿上传件，矢量 + 300dpi 位图

修订记录
| 修订日期 | 轮次 | 改了什么 | 为什么改 |
|---|---|---|---|
| 2026-08-28 | 1 | 新建，取代 B0figures_v18.py 的 fig_ga | 旧版两 panel 都是数据曲线，看不出框架 |
| 2026-08-28 | 2 | 路线由 FigureSpec/SVG 改为 HTML+SVG→Chrome | ryan 判「太丑」；灰框流程图无视觉焦点 |
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# 发布树里 04outputs/ 叫 outputs/、01manuscript/figs/ 叫 figures/。两套名字都要能跑——
# README 的复现命令是在发布树里执行的，B0figures 之前正是在这里崩过。
_OUT_PUB = os.path.join(HERE, "..", "outputs")
OUT = _OUT_PUB if os.path.isdir(_OUT_PUB) else os.path.join(HERE, "..", "04outputs")
_FIG_PUB = os.path.join(HERE, "..", "figures")
FIGS = _FIG_PUB if os.path.isdir(_FIG_PUB) else os.path.join(
    HERE, "..", "06doc", "01manuscript", "figs")
STEM = "B1graphabs"

CANVAS_W, CANVAS_H = 1400, 560          # 2.50:1；@300dpi → 4375×1750 px，远超 Elsevier 下限


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
CEIL_1 = ICC
CEIL_5 = VAR_APPLE / (VAR_APPLE + VAR_FACE / 5.0)

# ── 配色：降饱和以适配大面积填充；深靛蓝做主，暖橙标噪声，红标天花板，绿标已验证 ──
INK, MUTED = "#16233A", "#5C6B84"
BLUE, ORANGE, RED, GREEN = "#2E5E8C", "#D9822B", "#B0392B", "#2E7D5B"


def ceiling_at(m: float) -> float:
    """m 次参考重复测定下的可达 R² 上界。"""
    return VAR_APPLE / (VAR_APPLE + VAR_FACE / m)


def build_html() -> str:
    # 上界曲线绘图区（局部坐标），y 轴 0.40–0.90
    cw, ch = 300, 150
    def px(m: float) -> float:
        return (m - 1) / 9 * cw

    def py(v: float) -> float:
        return ch - (v - 0.40) / 0.50 * ch

    curve = " ".join(f'{"M" if i == 0 else "L"}{px(m):.1f},{py(ceiling_at(m)):.1f}'
                     for i, m in enumerate(range(1, 11)))
    yticks = "".join(
        f'<g><line x1="-5" y1="{py(v):.1f}" x2="0" y2="{py(v):.1f}" stroke="#CFD6E0"/>'
        f'<text x="-9" y="{py(v) + 4:.1f}" font-size="11.5" fill="{MUTED}" '
        f'text-anchor="end">{v:.1f}</text></g>' for v in [0.5, 0.6, 0.7, 0.8, 0.9])
    xticks = "".join(
        f'<text x="{px(m):.1f}" y="{py(0.40) + 19:.1f}" font-size="11.5" fill="{MUTED}" '
        f'text-anchor="middle">{m}</text>' for m in [1, 3, 5, 7, 10])

    # 衰减比区间图（局部坐标），轴 0.45–0.62
    aw = 300
    def ax(v: float) -> float:
        return (v - 0.45) / 0.17 * aw

    aticks = "".join(
        f'<g><line x1="{ax(v):.1f}" y1="24" x2="{ax(v):.1f}" y2="29" stroke="#CFD6E0"/>'
        f'<text x="{ax(v):.1f}" y="45" font-size="11.5" fill="{MUTED}" '
        f'text-anchor="middle">{v:.2f}</text></g>' for v in [0.45, 0.50, 0.55, 0.60])

    # 苹果示意：5 个等角分布的测定点，透明度不同以暗示测值彼此不一致
    dots = "".join(
        f'<circle cx="{60 + 26 * math.cos(a):.1f}" cy="{60 + 26 * math.sin(a):.1f}" '
        f'r="6.5" fill="{ORANGE}" opacity="{o}"/>'
        for a, o in zip([-1.5708, -0.3142, 0.9425, 2.1991, 3.4558],
                        ["1", ".62", ".86", ".5", ".74"], strict=True))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Graphical abstract</title>
<meta name="hz:slide-selector" content=".infographic">
<meta name="hz:canvas-width" content="{CANVAS_W}">
<meta name="hz:canvas-height" content="{CANVAS_H}">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#fff}}
/* 不给 @page 定尺寸，Chrome --print-to-pdf 会按 Letter 出并把右栏裁掉 */
@page{{size:{CANVAS_W}px {CANVAS_H}px;margin:0}}
@media print{{body{{margin:0}}.infographic{{page-break-after:avoid}}}}
.infographic{{position:relative;width:{CANVAS_W}px;height:{CANVAS_H}px;background:#fff;
  font-family:"Avenir Next","Helvetica Neue",Helvetica,Arial,sans-serif;
  color:{INK};overflow:hidden}}
.hdr{{position:absolute;left:0;top:0;width:{CANVAS_W}px;height:78px;background:{INK};
  display:flex;align-items:center;padding:0 44px}}
.hdr h1{{font-size:27px;font-weight:600;color:#fff;letter-spacing:-.2px}}
.hdr .em{{color:#F2B872}}
.col{{position:absolute;top:112px}}
.step{{font-size:12px;font-weight:700;letter-spacing:1.6px;color:{MUTED};margin-bottom:9px}}
.h2{{font-size:19px;font-weight:600;margin-bottom:13px;line-height:1.28}}
.p{{font-size:14.5px;line-height:1.52;color:{MUTED}}}
.big{{font-size:54px;font-weight:600;letter-spacing:-2px;color:{RED};line-height:1}}
.big .u{{font-size:26px;font-weight:500;letter-spacing:-.3px}}
.sub{{font-size:14px;color:{MUTED};margin-top:8px;line-height:1.5}}
.rule{{position:absolute;top:112px;width:1px;height:352px;background:#DDE2EA}}
.foot{{position:absolute;left:0;bottom:0;width:{CANVAS_W}px;height:62px;background:#F4F6F9;
  display:flex;align-items:center;justify-content:center;border-top:1px solid #E3E7EE}}
.foot span{{font-size:17px;font-weight:500;color:{INK}}}
.tool{{margin-bottom:26px}}
.tool .t{{font-size:15.5px;font-weight:600;margin-bottom:4px}}
.tool .d{{font-size:13.5px;color:{MUTED};line-height:1.55}}
text{{font-family:"Avenir Next","Helvetica Neue",Helvetica,Arial,sans-serif}}
</style></head>
<body><div class="infographic" data-canvas-width="{CANVAS_W}" data-canvas-height="{CANVAS_H}">

<div class="hdr"><h1>Calibration R&sup2; is capped by the reference,
  <span class="em">not by the model</span></h1></div>

<div class="rule" style="left:466px"></div>
<div class="rule" style="left:933px"></div>

<div class="col" style="left:44px;width:382px">
  <div class="step">THE UNCHECKED PREMISE</div>
  <div class="h2">One fruit, five destructive<br>determinations &mdash; they disagree.</div>
  <svg width="382" height="120" viewBox="0 0 382 120">
    <circle cx="60" cy="60" r="42" fill="#FBF0E2" stroke="{ORANGE}" stroke-width="2"/>
    {dots}
    <text x="122" y="44" font-size="14.5" fill="{MUTED}">reference value &#563;</text>
    <text x="122" y="66" font-size="14.5" fill="{MUTED}">carries its own</text>
    <text x="122" y="88" font-size="15" font-weight="600" fill="{ORANGE}">measurement noise</text>
  </svg>
  <div class="p" style="margin-top:6px">Yet R&sup2; and RMSEP are read as if &#563; were
  exact. In {N_FRUIT:,} apples the reference&rsquo;s own scatter is
  <b style="color:{ORANGE}">{WITHIN:.2f}%</b> of the total.</div>
  <svg width="382" height="52" viewBox="0 0 382 52" style="margin-top:14px">
    <rect x="0" y="8" width="{382 * BETWEEN / 100:.1f}" height="26" fill="{GREEN}" opacity=".85"/>
    <rect x="{382 * BETWEEN / 100:.1f}" y="8" width="{382 * WITHIN / 100:.1f}" height="26"
      fill="{ORANGE}" opacity=".85"/>
    <text x="6" y="26" font-size="13" font-weight="600" fill="#fff">between-fruit {BETWEEN:.2f}%</text>
    <text x="{382 * BETWEEN / 100 + 7:.1f}" y="26" font-size="13" font-weight="600"
      fill="#fff">within-fruit {WITHIN:.2f}%</text>
    <text x="0" y="48" font-size="12" fill="{MUTED}">signal</text>
    <text x="382" y="48" font-size="12" fill="{MUTED}" text-anchor="end">noise in the reference</text>
  </svg>
</div>

<div class="col" style="left:510px;width:382px">
  <div class="step">THE ERROR BUDGET</div>
  <div class="h2">That noise sets a ceiling no<br>model can pass.</div>
  <div class="big">{CEIL_1:.3f}<span class="u"> max R&sup2;</span></div>
  <div class="sub">against a single-face reference; <b style="color:{INK}">{CEIL_5:.3f}</b>
  against the five-face mean. Not fitted &mdash; it follows from the variance split alone.</div>
  <svg width="352" height="162" viewBox="-42 -10 {cw + 64} {ch + 52}" style="margin-top:14px">
    <line x1="0" y1="{py(0.40)}" x2="{cw}" y2="{py(0.40)}" stroke="#CFD6E0"/>
    <line x1="0" y1="0" x2="0" y2="{py(0.40)}" stroke="#CFD6E0"/>
    {yticks}
    <path d="{curve}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>
    <circle cx="{px(1):.1f}" cy="{py(CEIL_1):.1f}" r="6" fill="{RED}"/>
    <circle cx="{px(5):.1f}" cy="{py(CEIL_5):.1f}" r="6" fill="{GREEN}"/>
    <text x="{px(1) + 11:.1f}" y="{py(CEIL_1) + 5:.1f}" font-size="12.5" font-weight="600"
      fill="{RED}">{CEIL_1:.3f}</text>
    <text x="{px(5) + 11:.1f}" y="{py(CEIL_5) + 5:.1f}" font-size="12.5" font-weight="600"
      fill="{GREEN}">{CEIL_5:.3f}</text>
    {xticks}
    <text x="{cw / 2:.0f}" y="{py(0.40) + 40:.0f}" font-size="12.5" fill="{MUTED}"
      text-anchor="middle">reference replicates per fruit</text>
  </svg>
</div>

<div class="col" style="left:977px;width:382px">
  <div class="step">WHAT YOU GET</div>
  <div class="tool">
    <div class="t">1 &nbsp;A design rule</div>
    <div class="d">Invert the ceiling: name a target R&sup2;, read off how many
    replicate determinations your reference needs.</div>
  </div>
  <div class="tool">
    <div class="t">2 &nbsp;A falsifiable diagnostic</div>
    <div class="d">Predicts the attenuation ratio with
    <b style="color:{INK}">no free parameter</b>, so it can be wrong.
    On {N_FRUIT:,} apples it was not.</div>
  </div>
  <svg width="382" height="150" viewBox="-14 -32 {aw + 34} 150" style="margin-top:4px">
    <line x1="0" y1="24" x2="{aw}" y2="24" stroke="#CFD6E0"/>
    {aticks}
    <line x1="{ax(ATT_LO):.1f}" y1="24" x2="{ax(ATT_HI):.1f}" y2="24" stroke="{GREEN}"
      stroke-width="7" stroke-linecap="round"/>
    <circle cx="{ax(ATT_OBS):.1f}" cy="24" r="6.5" fill="{GREEN}"/>
    <text x="{ax(ATT_OBS):.1f}" y="8" font-size="13" font-weight="600" fill="{GREEN}"
      text-anchor="middle">measured {ATT_OBS:.3f}</text>
    <text x="{ax(ATT_OBS):.1f}" y="-9" font-size="11.5" fill="{MUTED}"
      text-anchor="middle">95% CI [{ATT_LO:.3f}, {ATT_HI:.3f}]</text>
    <line x1="{ax(ATT_PRED):.1f}" y1="12" x2="{ax(ATT_PRED):.1f}" y2="36" stroke="{INK}"
      stroke-width="2" stroke-dasharray="3 2"/>
    <text x="{ax(ATT_PRED):.1f}" y="63" font-size="12" font-weight="600" fill="{INK}"
      text-anchor="middle">predicted {ATT_PRED:.4f}</text>
    <text x="{ax(ATT_PRED):.1f}" y="79" font-size="11.5" fill="{MUTED}"
      text-anchor="middle">inside the interval &#10003;</text>
    <line x1="{ax(CEIL_1):.1f}" y1="12" x2="{ax(CEIL_1):.1f}" y2="36" stroke="{RED}"
      stroke-width="2" stroke-dasharray="3 2"/>
    <text x="{ax(CEIL_1):.1f}" y="63" font-size="12" font-weight="600" fill="{RED}"
      text-anchor="middle">{CEIL_1:.3f}</text>
    <text x="{ax(CEIL_1):.1f}" y="79" font-size="11.5" fill="{MUTED}"
      text-anchor="middle">excluded</text>
    <text x="{aw / 2:.0f}" y="112" font-size="12.5" fill="{MUTED}"
      text-anchor="middle">attenuation ratio &mdash; measured vs. predicted</text>
  </svg>
</div>

<div class="foot"><span>Report performance against a <b>reference-limited ceiling</b>
  &mdash; not against unity.</span></div>
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
    print(f"  ICC={ICC:.4f}  果间={BETWEEN:.2f}%  果内={WITHIN:.2f}%  "
          f"上界 单面={CEIL_1:.3f} 五面均值={CEIL_5:.3f}  n={N_FRUIT}")
    print(f"  衰减比实测 {ATT_OBS:.4f} [{ATT_LO:.4f}, {ATT_HI:.4f}]  预言 {ATT_PRED:.4f}")

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
