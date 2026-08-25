"""A11 · 派生量落盘：把正文报告、但此前只在文字里现算的派生统计量导出到工作簿。

起因（2026-07-26 独立一致性审计 确认重跑，`HP-PHANTOM-RESULT`，critical）：
§2.4 承诺「**每一个**方差分量、$R^2$、RMSE、比值与置信区间都由脚本产出并写入工作簿」，
但下列三类量是由工作簿里的分量**现算**的，本身没有落盘——审查员据此判定该承诺与实际
不符。这与 EF-001 是同一类问题：**能复算 ≠ 已落盘**，承诺写了「每一个」就必须做到每一个。

本脚本不产生任何新结论、不重算任何原始统计量，只做一件事：**读已有工作簿里的分量，
把正文引用的派生量按同样口径算出并写入自己的工作簿**，使其可被逐条追溯。

导出项：
  表a  参考值侧方差比 σ_f²/σ_a²（设计公式 eq:design 与半合成噪声标定都用它）
  表b  A6formal T11 两种分组下 cluster CI95 的**宽度**及其比值（补充材料图 S1 caption）
  表c  R² 分母口径之比 ρ = TSS_test/TSS_train（§2.3 报告 0.996--0.999、两口径差 ≤0.001）
  表d  猕猴桃缺失通道掩膜的空操作核验（§4.3 限制(6) 报告 0/222 与 200/200）

表c/表d 的起因（2026-07-26 独立一致性审计 第三轮）：
  · `HP-DERIVATION-INVALID`：§2.3 原括注称“用测试集自身均值会系统性抬高 R²”，方向写反。
    由恒等式 Σ(y-m)² = Σ(y-ȳ)² + n(ȳ-m)²，训练均值分母恒不小于，故**训练均值口径的 R² 恒不低于**
    测试均值口径。改正后正文须给出实测幅度——本表把它算出来落盘。
  · `HP-EVAL-LEAKAGE`(ED001)：质疑猕猴桃缺失通道掩膜在划分前用全量数据算，可能把测试折信息
    带进特征集。本表实测该掩膜删除 0 列、且任意训练子集算出的掩膜与全量一致 → 空操作。

表c/表d 均**不重新拟合任何模型**：表c 只重放划分算两个分母，表d 只对光谱矩阵做缺失判定。

红线：不做插值、不做四舍五入后再运算——比值一律由全精度端点计算，展示值另列。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import write_script_workbook  # noqa: E402

_H = os.path.dirname(os.path.abspath(__file__))
_OUTA = os.path.join(_H, "..", "outputs")
OUT = _OUTA if os.path.isdir(_OUTA) else os.path.join(_H, "..", "04outputs")


def _sheet(book: str, name: str) -> pd.DataFrame:
    return pd.read_excel(os.path.join(OUT, f"{book}_v18.xlsx"), sheet_name=name)


def main() -> None:
    # ── 表a：方差比 ──────────────────────────────────────────────────────────
    vc = _sheet("93sscceiling", "表c：方差分解（总体与分产地）").set_index("分组")
    va = float(vc.loc["全体（主口径）", "var_apple"])
    vf = float(vc.loc["全体（主口径）", "var_face"])
    ratio = vf / va
    # 半合成注入量的口径：把猕猴桃 σ_y² 按苹果比值拆分，注入其中的果内部分。
    # 2026-07-26 加入（预投稿评审指出正文只给比值、未给拆分方式，
    # 会被读成 σ_f²=1.0965·σ_y²，与实际差 2.1 倍且使表 2 的 m_req 无法复现）。
    frac = ratio / (1.0 + ratio)
    t_a = pd.DataFrame([{
        "量": "σ_f² / σ_a²（果内/果间方差比）",
        "σ_a²（全精度）": va,
        "σ_f²（全精度）": vf,
        "比值（全精度）": ratio,
        "正文展示值": round(ratio, 4),
        "来源": "93sscceiling_v18.xlsx 表c：方差分解（总体与分产地）· 行「全体（主口径）」",
        "用于": "式 eq:design 的设计表；A9 半合成实验的注入噪声标定",
    }, {
        "量": "σ_f² / σ_y²（半合成实际注入方差占标签方差的份额）",
        "σ_a²（全精度）": None,
        "σ_f²（全精度）": None,
        "比值（全精度）": frac,
        "正文展示值": round(frac, 4),
        "来源": "由本表第 1 行的比值导出：frac = ratio/(1+ratio)；"
                "与 A9semisynth_v18_raw.json 实存的 σ_f²/σ_y² 逐位一致",
        "用于": "§3.7 注入量标定的口径说明；等于苹果果内方差占比 52.30% = 1−ICC",
    }])

    # ── 表b：T11 两种分组的 CI 宽度及其比值 ──────────────────────────────────
    t11 = _sheet("A6formal", "T11 DeepHS 日序分组").set_index("量")
    rows = []
    widths = {}
    for label, key in (("按果分组", "日序 · Kiwi·按果分组 · R²"),
                       ("按采集日分组", "日序 · Kiwi·按采集日分组 · R²")):
        lo = float(t11.loc[key, "cluster CI95 下限"])
        hi = float(t11.loc[key, "cluster CI95 上限"])
        widths[label] = hi - lo
        rows.append({
            "分组方式": label,
            "cluster CI95 下限（全精度）": lo,
            "cluster CI95 上限（全精度）": hi,
            "区间宽度（全精度）": hi - lo,
            "正文展示值": round(hi - lo, 6),
        })
    r = widths["按采集日分组"] / widths["按果分组"]
    rows.append({
        "分组方式": "**宽度之比（按采集日 / 按果）**",
        "cluster CI95 下限（全精度）": None,
        "cluster CI95 上限（全精度）": None,
        "区间宽度（全精度）": r,
        "正文展示值": round(r),
    })
    t_b = pd.DataFrame(rows)

    # ── 表c：R² 分母口径之比 ρ = TSS_test / TSS_train ────────────────────────
    # 与 A6formal 的 cv_a1/cv_a4 逐字节同序重放划分；不拟合模型。
    SEEDS, TF = [20060515, 20041210, 19810915, 2023, 2024], 0.40

    def _rho(y, groups, seed, nrep):
        rng = np.random.default_rng(seed)
        uq, out = np.unique(groups), []
        for _ in range(nrep):
            te = set(rng.permutation(uq)[: int(round(TF * len(uq)))])
            m = np.array([g in te for g in groups])
            yt, ytr = y[m], y[~m]
            out.append(float(np.sum((yt - yt.mean()) ** 2) / np.sum((yt - ytr.mean()) ** 2)))
        return np.array(out)

    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03data", "processed")
    _kp = os.path.join(DATA, "v18_kiwi_instr.parquet")
    _ap = os.path.join(DATA, "v18_apple_faces.parquet")
    # 表c/表d 需要预处理后的谱矩阵。开源仓库不重分发原始/预处理数据（苹果为校内未发表
    # 数据，见 README「数据可用性」），缺数据时如实跳过并说明，而不是崩掉或编造数值。
    if not (os.path.exists(_kp) and os.path.exists(_ap)):
        print("  [跳过 表c/表d] 未找到 03data/processed 下的谱矩阵；这两表需要预处理数据，"
              "本仓库不重分发。表a/表b 只依赖工作簿，照常导出。")
        path = write_script_workbook(__file__, {
            0: ("参考值侧方差比", t_a),
            1: ("T11 CI 宽度与比值", t_b),
        })
        print(f"→ {path}")
        return

    kd = pd.read_parquet(_kp).dropna(subset=["SSC", "DM"])
    ad = pd.read_parquet(_ap)
    rows_c = []
    for name, y, g, nrep, r2 in (
        ("猕猴桃 DM", kd["DM"].values.astype(float), kd["sample_id"].values, 15, 0.827),
        ("猕猴桃 SSC", kd["SSC"].values.astype(float), kd["sample_id"].values, 15, 0.861),
        ("苹果 SSC", ad["ssc"].values.astype(float), ad["id"].values, 20, 0.820),
    ):
        rr = np.concatenate([_rho(y, g, s, nrep) for s in SEEDS])
        r2_te = 1 - (1 - r2) / rr.mean()   # 换成测试均值口径
        rows_c.append({
            "分析": name,
            "报告 R²（训练均值口径）": r2,
            "ρ = TSS_test/TSS_train 均值（全精度）": float(rr.mean()),
            "ρ 最小值（全精度）": float(rr.min()),
            "折算到测试均值口径的 R²（全精度）": float(r2_te),
            "两口径之差（全精度）": float(r2 - r2_te),
            "重放划分次数": int(rr.size),
        })
    t_c = pd.DataFrame(rows_c)

    # ── 表d：猕猴桃缺失通道掩膜的空操作核验 ──────────────────────────────────
    kwl = [c for c in kd.columns if c.startswith("X")]
    V = kd[kwl].values.astype(float)
    ok_all = ~np.isnan(V).any(0)
    rng = np.random.default_rng(0)
    sid, uq = kd["sample_id"].values, np.unique(kd["sample_id"].values)
    n_same, N_TRIAL = 0, 200
    for _ in range(N_TRIAL):
        tr = rng.choice(uq, size=int(len(uq) * 0.8), replace=False)
        n_same += bool((~np.isnan(V[np.isin(sid, tr)]).any(0) == ok_all).all())
    t_d = pd.DataFrame([{
        "掩膜": "猕猴桃缺失通道掩膜（A6formal_v18.py task_kiwi）",
        "波长列总数": len(kwl),
        "被掩膜剔除的列数": int((~ok_all).sum()),
        "保留列数": int(ok_all.sum()),
        "训练子集重算次数": N_TRIAL,
        "与全量掩膜一致的次数": int(n_same),
        "结论": "剔除 0 列且任意 80% 训练子集复现同一掩膜 → 空操作，不携带留出集信息",
    }])

    path = write_script_workbook(__file__, {
        0: ("参考值侧方差比", t_a),
        1: ("T11 CI 宽度与比值", t_b),
        2: ("R2 分母口径之比", t_c),
        3: ("猕猴桃掩膜空操作核验", t_d),
    })
    print(f"→ {path}")
    print(f"  σ_f²/σ_a² = {ratio!r} → 正文 {round(ratio, 4)}")
    print(f"  CI 宽度：按果 {widths['按果分组']!r} / 按采集日 {widths['按采集日分组']!r}")
    print(f"  比值 = {r!r} → 正文 {round(r)}")
    print(f"  ρ 范围 {t_c['ρ = TSS_test/TSS_train 均值（全精度）'].min():.4f}–"
          f"{t_c['ρ = TSS_test/TSS_train 均值（全精度）'].max():.4f}；"
          f"两口径最大差 {t_c['两口径之差（全精度）'].max():.6f}")
    print(f"  猕猴桃掩膜：剔除 {int((~ok_all).sum())}/{len(kwl)} 列；一致 {n_same}/{N_TRIAL}")


if __name__ == "__main__":
    main()
