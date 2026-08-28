# 0NFIGURE_AUDIT — 图表审计（v18 / C4′，2026-07-25）

> 图表规范：每张图同时产出中文版与英文版，数据、配色、字号、版式完全一致，只切文本。
> 生成脚本：`code/B0figures_v18.py`（单一入口，`save_figure_bilingual` 统一保存）。
> 红线：**每张图的每个数字都必须能追到 `outputs/*.xlsx`，绘图脚本里的常量不算证据**
> （EF-001 教训）。

## 一、清单

| 图 | 中文件名 | 英文件名 | 面板 | 数据来源 | 取数方式 |
|---|---|---|---|---|---|
| a | `B0figures_v18-图a.pdf` | `-fig-a.pdf` | 3 | `93sscceiling_v18.xlsx` 表c/表d | **现场读取** ✅ |
| b | `B0figures_v18-图b.pdf` | `-fig-b.pdf` | 1 | `A1consolidate` 表a/b/c ＋ `A5aggregate` 表b | **现场读取** ✅ |
| c | `B0figures_v18-图c.pdf` | `-fig-c.pdf` | 2 | `97instrbudget_v18.xlsx` 表d | **现场读取** ✅ |
| d | `B0figures_v18-图d.pdf` | `-fig-d.pdf` | **1**（原 2） | `A4formal` 表c | **现场读取** ✅ |
| **S1** | `B0suppfig_v18-图a.pdf` | `-fig-a.pdf` | 1 | `A6formal` T11 | **现场读取** ✅ **补充材料** |
| **GA** | `B1graphabs.pdf` / `.html`（单语，英文） | — | — | `93sscceiling` 表c ＋ `A5aggregate` 表b | **现场读取** ✅ **图文摘要**（不走 matplotlib，见下 G2） |
| **e** | `B0figures_v18-图e.pdf` | `-fig-e.pdf` | 2 | `A9semisynth_v18.xlsx` 退化轨迹 | **现场 `pd.read_excel` 读取，零硬编码** ✅ |

## 二、逐项合规

| 项 | 要求 | 状态 |
|---|---|---|
| F1 | 中英双版，**数据/配色/字号/版式完全一致，仅文本切换** | ✅ 7×2 = 14 个 PDF（正文 5 + 补充 1 + 图形摘要 1），均由同一 `plot_fn(lang)` 产出 |
| F2 | 矢量 PDF，300 dpi savefig | ✅ `savefig.dpi=300`，`format="pdf"` |
| F3 | 色盲友好调色板 | ✅ Okabe–Ito（`#0072B2/#E69F00/#009E73/#D55E00/#CC79A7`） |
| F4 | 面板标签 (a)(b)(c) | ✅ `_panel_tag()` 统一 |
| F5 | 误差棒必须注明含义 | ✅ 图 a(c) ICC 自助区间；图 d/e cluster bootstrap 95% CI，caption 均写明 |
| F6 | 全部被正文 `\ref` 引用 | ✅ fig:a–fig:e 各 1 次；fig:s1 被补充材料引 1 次；图形摘要按 Elsevier 规程单独上传、不在正文引用（不计悬空） |
| F7 | caption 与内容一致 | ✅ claim audit R7 逐图核对，5 条 caption 全部 `exact_match`/`rounding_ok` |
| F8 | 两个语言版本的图不混用 | ✅ 中文版引 `-图x.pdf`，英文版引 `-fig-x.pdf`，脚本扫描确认 |
| F9 | figs/ 目录为实拷贝而非软链 | ✅ `cp` 实拷贝（软链会让审计脚本的 `find -type f` 返回假零） |

## 三、图 e 的取数实现（新增，本轮）

```python
d = pd.read_excel(os.path.join(OUT, "A9semisynth_v18.xlsx"), sheet_name="退化轨迹")
d = d[pd.to_numeric(d["m（参考重复次数）"], errors="coerce").notna()].copy()   # 滤掉 "∞（干净基线）" 行
...
ax1.plot(m, g["V9b 结构上界（正文口径）"], ":")     # 上界
ax1.plot(m, g["预言 R²_pred"], "-")                # 预言
ax1.errorbar(m, obs, yerr=[obs-lo, hi-obs], ...)   # 实测 + cluster CI
ax2.plot(m, g["相对偏差 obs/pred−1"], ...)          # 相对偏差
```

四条曲线全部取自工作簿列，**绘图脚本不重算任何统计量**——包括 cluster bootstrap CI
（若在绘图脚本里用 `np.std` 重算，会得到与正文不同型的区间，正是 EF-001 那类隐患）。

## 四、遗留

| 项 | 状态 | 说明 |
|---|---|---|
| ~~G1 图 a–d 仍用脚本常量~~ | ✅ **已闭合（2026-07-26）** | 独立一致性审计 判 `HP-PHANTOM-RESULT`（major）——绘图脚本的字面量与 §2.4「每个统计量均由脚本产出并写入工作簿」的声明矛盾，即 EF-001 复发。已把图 a–d 的全部数值改为从工作簿现场读取，脚本内**零字面量常量赋值**，读不到直接抛错、不回退硬编码。重生成后所有数值与原值一致（且为全精度） |
| ~~G2 Graphical abstract~~ | ✅ **已重做（2026-08-28）** | 由 `B1graphabs.py` 产出，1050×420 pt = 4200×1680 px @300 dpi、比例 2.50:1，满足 Elsevier「最小 531×1328 px (h×w)」；矢量 PDF，字体全为拉丁字体（分栏标签用 A./B./C. 而非 ①②③——后者 Helvetica 缺字，转 PDF 会回退到中文字体）。**画的是论证骨架不是结果曲线**：A 被忽略的前提 → B 拆成误差预算 → C 两个可用工具。数值现场读 `93sscceiling` 表c 与 `A5aggregate` 表b。技术路线：Python 现场读工作簿 → 自包含 HTML（内联 SVG）→ Chrome headless 出 PDF/PNG。HTML 随产物留档（`figures/B1graphabs.html`），任何浏览器打开即可核对与改版式。**替换了原 2026-07-26 的 `fig_ga()`**——那一版是图 a(c) 与图 e(a) 的重新拼版，两个 panel 都是数据曲线，读者不读正文看不懂框架 |
| **G3 §3.5 时序半段移入补充材料** | ✅ **已做（2026-07-26）** | 依预投稿评审建议（正文承载 5 条线、缺单一中心）。搬走 593 词 / 1439 字 + 图 d 面板 (b) → `Supplementary_{en,zh}.tex` §S1 + 图 S1。图 d 由 2 面板降为 1 面板。**数字守恒实测：正文＋补充合计 252 个不同数值，中英零单边** |
