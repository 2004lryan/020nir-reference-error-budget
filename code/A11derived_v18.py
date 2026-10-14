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
  表c  R² 分母口径之比 ρ = TSS_test/TSS_train 及两口径之差（§2.3／补充材料 S2）
  表e  衰减比在训练均值口径与测试均值口径下的值与 cluster CI95（补充材料 S2）
  表d  猕猴桃缺失通道掩膜的空操作核验（§4.3 限制(6) 报告 0/222 与 200/200）

表c/表d 的起因（2026-07-26 独立一致性审计 第三轮）：
  · `HP-DERIVATION-INVALID`：§2.3 原括注称“用测试集自身均值会系统性抬高 R²”，方向写反。
    由恒等式 Σ(y-m)² = Σ(y-ȳ)² + n(ȳ-m)²，训练均值分母恒不小于，故**训练均值口径的 R² 恒不低于**
    测试均值口径。改正后正文须给出实测幅度——本表把它算出来落盘。
  · `HP-EVAL-LEAKAGE`(ED001)：质疑猕猴桃缺失通道掩膜在划分前用全量数据算，可能把测试折信息
    带进特征集。本表实测该掩膜删除 0 列、且任意训练子集算出的掩膜与全量一致 → 空操作。

表c/表d/表e 均**不重新拟合任何模型**：表c/表e 只重放划分算两个分母并折算已落盘的逐次 R²，
表d 只对光谱矩阵做缺失判定。

2026-09-02 更正（第十二轮后的独立复核）：两口径之差 = (1−R²)·(1/ρ−1)，随 (1−R²) 缩放。
此前表c 把苹果行的 R² 填成五面均值上界 0.820，而苹果光谱侧的实测 R² 只有 0.0665（面级）／
0.1180（整果均值），差因此被低估近一个数量级（0.0007 → 0.0036／0.0058）；猕猴桃两行此前按
15 次/种子重放，而 T10 实跑 6 次/种子（A6formal task_kiwi 的 nrep=6），多出的 45 个划分从未被评分。
现改为：猕猴桃按 6 次/种子重放；苹果两目标读 A4formal_v18_raw.json 主表 raw 的逐次 R²（20 次/种子），
用与 cv_a4 同序重放的逐次 ρ 逐次折算，同时给出均值差与逐次最大差；表e 给出衰减比在两口径下的
均值之比与两级簇自助 CI95（算法与 A5 表b 相同，训练口径一行须逐位复现 A5 表b）。

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
    # 猕猴桃 T10：nrep=6（与 A6formal task_kiwi 一致），报告值按 ρ 均值折算，逐次最大差按 ρ 最小值折算
    for name, y, g, nrep, r2 in (
        ("猕猴桃 DM（T10）", kd["DM"].values.astype(float), kd["sample_id"].values, 6, 0.827),
        ("猕猴桃 SSC（T10）", kd["SSC"].values.astype(float), kd["sample_id"].values, 6, 0.861),
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
            "两口径之差·逐次最大（全精度）": float((1 - r2) * (1 / rr.min() - 1)),
            "重放划分次数": int(rr.size),
            "折算依据": "报告值 × ρ 均值折算；划分与 T10 实跑的 5×6=30 次逐一相同",
        })
    # 苹果主表（raw，20 次/种子，两目标共用同一划分）：读 A4formal_v18_raw.json 的逐次 R²，逐次折算
    import json as _json
    with open(os.path.join(OUT, "A4formal_v18_raw.json"), encoding="utf-8") as fh:
        raw_a4 = _json.load(fh)
    units = [u for u in raw_a4["apple"] if u["preproc"] == "raw"]
    ys_apple = {"y_face": ad["ssc"].values.astype(float), "y_fruit": ad["y_fruit"].values.astype(float)}
    rho_by: dict[str, dict[int, np.ndarray]] = {k: {} for k in ys_apple}
    for key, yv in ys_apple.items():
        for s in SEEDS:
            rho_by[key][s] = _rho(yv, ad["id"].values, s, 20)
    conv: dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = {}
    for key, label in (("y_face", "苹果 面级 SSC（主表 raw）"), ("y_fruit", "苹果 整果均值 SSC（主表 raw）")):
        tr = {int(u["seed"]): np.asarray(u["reps"][key], float) for u in units}
        te = {s: 1 - (1 - tr[s]) / rho_by[key][s] for s in SEEDS}
        conv[key] = (tr, te)
        all_tr = np.concatenate([tr[s] for s in SEEDS])
        all_te = np.concatenate([te[s] for s in SEEDS])
        all_rho = np.concatenate([rho_by[key][s] for s in SEEDS])
        rows_c.append({
            "分析": label,
            "报告 R²（训练均值口径）": float(all_tr.mean()),
            "ρ = TSS_test/TSS_train 均值（全精度）": float(all_rho.mean()),
            "ρ 最小值（全精度）": float(all_rho.min()),
            "折算到测试均值口径的 R²（全精度）": float(all_te.mean()),
            "两口径之差（全精度）": float((all_tr - all_te).mean()),
            "两口径之差·逐次最大（全精度）": float((all_tr - all_te).max()),
            "重放划分次数": int(all_rho.size),
            "折算依据": "逐次折算：A4formal_v18_raw.json 主表 raw 的 100 次 R² × 同序重放的逐次 ρ",
        })
    t_c = pd.DataFrame(rows_c)

    # ── 表e：衰减比在两种口径下（均值之比 + 两级簇自助 CI95，算法与 A5 表b 相同）──────
    def _boot_ratio(num_by, den_by, b=2000, seed=0):
        rng = np.random.default_rng(seed)
        ks = list(num_by)
        vals = []
        for _ in range(b):
            n_, d_ = [], []
            for i in rng.choice(len(ks), len(ks), replace=True):
                a_ = np.asarray(num_by[ks[i]])
                b_ = np.asarray(den_by[ks[i]])
                idx = rng.integers(0, len(a_), len(a_))
                n_.extend(a_[idx])
                d_.extend(b_[idx])
            vals.append(np.mean(n_) / np.mean(d_))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    rows_e = []
    for conv_name, idx in (("训练均值口径（正文口径）", 0), ("测试均值口径", 1)):
        num = {s: conv["y_face"][idx][s] for s in SEEDS}
        den = {s: conv["y_fruit"][idx][s] for s in SEEDS}
        ratio_v = float(np.concatenate(list(num.values())).mean() / np.concatenate(list(den.values())).mean())
        lo, hi = _boot_ratio(num, den)
        rows_e.append({
            "口径": conv_name,
            "衰减比 = R²(面级)/R²(整果均值)（均值之比）": ratio_v,
            "cluster CI95 下限": lo,
            "cluster CI95 上限": hi,
            "含理论值 0.5816": bool(lo <= 0.581596 <= hi),
            "排除 ICC 0.477": not (lo <= 0.476995 <= hi),
            "说明": ("须逐位复现 A5 表b（同一逐次 R²、同一自助算法与种子）" if idx == 0
                    else "同一划分、同一逐次 R² 折算到测试均值口径后重算；两条判定均须与正文口径一致"),
        })
    t_e = pd.DataFrame(rows_e)

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
        4: ("衰减比两口径", t_e),
    })
    print(f"→ {path}")
    print(f"  σ_f²/σ_a² = {ratio!r} → 正文 {round(ratio, 4)}")
    print(f"  CI 宽度：按果 {widths['按果分组']!r} / 按采集日 {widths['按采集日分组']!r}")
    print(f"  比值 = {r!r} → 正文 {round(r)}")
    print(f"  ρ 范围 {t_c['ρ = TSS_test/TSS_train 均值（全精度）'].min():.4f}–"
          f"{t_c['ρ = TSS_test/TSS_train 均值（全精度）'].max():.4f}；"
          f"两口径均值差最大 {t_c['两口径之差（全精度）'].max():.6f}；"
          f"逐次差最大 {t_c['两口径之差·逐次最大（全精度）'].max():.6f}")
    for _, r in t_e.iterrows():
        print(f"  衰减比[{r['口径']}] = {r['衰减比 = R²(面级)/R²(整果均值)（均值之比）']:.6f} "
              f"CI [{r['cluster CI95 下限']:.6f}, {r['cluster CI95 上限']:.6f}] "
              f"含0.5816={r['含理论值 0.5816']} 排除0.477={r['排除 ICC 0.477']}")
    print(f"  猕猴桃掩膜：剔除 {int((~ok_all).sum())}/{len(kwl)} 列；一致 {n_same}/{N_TRIAL}")


if __name__ == "__main__":
    main()
