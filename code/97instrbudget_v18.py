"""97 · 012 猕猴桃：仪器间分量可辨识性体检（C4′ 第 3 根支柱的决定性预实验）

C4′ 主张的误差预算有两个分量：
  (i) 参考值侧的果内变异 —— 001 苹果可估（5 面各自破坏性测定）；
  (ii) 测量侧的仪器间重复性 —— 声称 012 可估（同果跨 5 台仪器重复扫描）。

本脚本只回答 (ii) 是否真的可辨识，以及它有多大。关键结构事实：012 每个 sample_id
的多条谱**各来自不同设备，且同一设备内没有技术重复**（n_scan == n_device 恒成立）。
因此 fruit×device 单元内无重复 ⇒ 「仪器×样本交互」与「纯扫描/复位噪声」**不可分离**，
只能估一个合并项。这与 001 侧「生物空间异质 vs 破坏性取样流程变异」不可分离是同构的，
必须同样如实命名，不得宣称估到了纯粹的"仪器重复性"。

产出：
  表a  设计结构与可辨识性判定
  表b  设备固定效应（偏倚/尺度）——crossed 设计下可辨识
  表c  谱层面方差分解：果间 / 设备主效应 / 果×设备合并残差
  表d  预测层面：同果跨设备的预测离散度（RMSEP 里属于"换台仪器就会变"的部分）
  表e  反演上界：由实测 R² 反推参考值可靠性下界
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

SEED = 2004
import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA = os.path.join(DATA_ROOT, "012_kiwifruit_nir_drymatter", "kiwifruit_dat.csv")
N_COMP = 10


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA, low_memory=False)
    wave = sorted([c for c in df.columns if c.startswith("X") and c[1:].isdigit()],
                  key=lambda c: int(c[1:]))
    df = df.dropna(subset=["sample_id", "device"])
    # 1068–1137 nm 这 24 个波段在**全部 5 台设备**上都缺（并非某台独有），
    # 保留会让 PLS 直接抛 NaN。取五台设备的公共可用区间 402–1065 nm（222 波段）。
    bad = [c for c in wave if df[c].isna().any()]
    if bad:
        wave = [c for c in wave if c not in bad]
    return df, wave


def snv(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def main() -> None:
    log = Logger(os.path.splitext(os.path.basename(__file__))[0])
    sys.stdout = log
    df, wave = load()

    # ---------- 表a：设计结构与可辨识性 ----------
    g = df.groupby("sample_id")
    n_scan = g.size()
    n_dev = g["device"].nunique()
    within_dev_rep = int((n_scan - n_dev).sum())
    multi = n_dev[n_dev >= 2].index
    lab_ok = df.dropna(subset=["DM"])["sample_id"].nunique()
    tab_a = pd.DataFrame([
        {"项": "总谱条数", "值": len(df)},
        {"项": "唯一 sample_id 数", "值": int(df["sample_id"].nunique())},
        {"项": "设备数", "值": int(df["device"].nunique())},
        {"项": "同一设备内的技术重复条数（n_scan − n_device 之和）", "值": within_dev_rep},
        {"项": "≥2 台设备扫过的果数（仪器分量的有效 n）", "值": int(len(multi))},
        {"项": "5 台设备全扫的果数", "值": int((n_dev == 5).sum())},
        {"项": "仅 1 台设备的果数（对仪器分量无贡献）", "值": int((n_dev == 1).sum())},
        {"项": "有 DM 标签的果数", "值": int(lab_ok)},
        {"项": "**可辨识性判定**",
         "值": ("设备主效应 可辨识；果×设备残差 可估但与扫描/复位噪声**不可分离**"
                if within_dev_rep == 0 else "单元内有重复，交互与噪声可分离")},
    ])

    sub = df[df["sample_id"].isin(multi)].copy()
    X = sub[wave].to_numpy(dtype=float)
    Xs = snv(X)

    # ---------- 表b/表c：谱层面方差分解（crossed, 不平衡） ----------
    # 用扣除果均值的方式估设备主效应，避免不平衡设计下的边际均值偏倚
    sub_fruit_mean = pd.DataFrame(Xs, index=sub["sample_id"].values).groupby(level=0).transform("mean")
    resid_fruit = Xs - sub_fruit_mean.to_numpy()
    dev_eff = pd.DataFrame(resid_fruit, index=sub["device"].values).groupby(level=0).mean()

    rows_b = []
    for d, vec in dev_eff.iterrows():
        v = vec.to_numpy(dtype=float)
        rows_b.append({
            "device": d,
            "n_谱": int((sub["device"] == d).sum()),
            "设备主效应 L2 范数": float(np.linalg.norm(v)),
            "设备主效应 均值偏移": float(v.mean()),
            "设备主效应 最大绝对值": float(np.abs(v).max()),
        })
    tab_b = pd.DataFrame(rows_b).sort_values("设备主效应 L2 范数", ascending=False)

    dev_map = dev_eff.reindex(sub["device"].values).to_numpy()
    resid_after_dev = resid_fruit - dev_map
    ss_total = float((Xs - Xs.mean(axis=0)) .var(axis=0).sum())
    ss_fruit = float((sub_fruit_mean.to_numpy() - Xs.mean(axis=0)).var(axis=0).sum())
    ss_dev = float(dev_map.var(axis=0).sum())
    ss_res = float(resid_after_dev.var(axis=0).sum())
    tab_c = pd.DataFrame([
        {"分量": "果间（生物 + 批次）", "方差和": ss_fruit, "占比%": 100 * ss_fruit / ss_total},
        {"分量": "设备主效应（系统偏倚，可校正）", "方差和": ss_dev, "占比%": 100 * ss_dev / ss_total},
        {"分量": "果×设备合并残差（交互 + 扫描/复位，**不可分离**）",
         "方差和": ss_res, "占比%": 100 * ss_res / ss_total},
        {"分量": "合计（SNV 后）", "方差和": ss_total, "占比%": 100.0},
    ])

    # ---------- 表d：预测层面的仪器间离散度 ----------
    # 关键：仪器间方差**不随模型变好而减小**（它是测量侧的），所以它占 MSEP 的比例
    # 随模型变好而**上升**。因此必须报告至少两个工作点，只报一个会误导。
    lab = sub.dropna(subset=["DM"]).copy()
    yl = lab["DM"].to_numpy(dtype=float)
    grp = lab["sample_id"].to_numpy()
    wl_nm = np.array([float(c[1:]) for c in wave])

    def cv_predict(Xl, n_comp=None):
        """返回 (逐样本 out-of-fold 预测, 以**训练折均值**为基准的总平方和)。

        SST 用训练折均值而非全样本均值 —— 与 docs/0NSTATISTICAL_PROTOCOL.md 及正文
        方法段声明的口径一致（不使用测试集自身信息）。此前本函数只按全样本均值算
        R²，与该声明不符（独立一致性审计 F001, critical）。
        """
        pred = np.full(len(lab), np.nan)
        sst_train = 0.0
        for tr, te in GroupKFold(n_splits=5).split(Xl, yl, groups=grp):
            if n_comp is None:                       # 训练折内再按果分组选成分数
                best, berr = 2, np.inf
                inner = list(GroupKFold(n_splits=4).split(Xl[tr], yl[tr], grp[tr]))
                for c in range(2, 26, 2):
                    e = [np.mean((PLSRegression(c).fit(Xl[tr][a], yl[tr][a])
                                  .predict(Xl[tr][b]).ravel() - yl[tr][b]) ** 2)
                         for a, b in inner]
                    if np.mean(e) < berr - 1e-9:
                        berr, best = np.mean(e), c
            else:
                best = n_comp
            pred[te] = PLSRegression(n_components=best).fit(Xl[tr], yl[tr]).predict(Xl[te]).ravel()
            sst_train += float(np.sum((yl[te] - yl[tr].mean()) ** 2))
        return pred, sst_train

    from scipy.signal import savgol_filter
    Xraw = lab[wave].to_numpy(dtype=float)
    configs = {
        f"A 弱模型（SNV 全谱，固定 {N_COMP} 成分）": (snv(Xraw), N_COMP),
        "B 强模型（一阶导 700–1065nm，CV 选成分）":
            (savgol_filter(Xraw[:, wl_nm >= 700], 11, 2, deriv=1, axis=1), None),
    }
    rows_d = []
    for cname, (Xl, nc) in configs.items():
        pred, sst_train = cv_predict(Xl, nc)
        s = pd.Series(pred, index=lab["sample_id"].values).groupby(level=0)
        spread, sd_within = s.max() - s.min(), s.std(ddof=1)
        rmsep = float(np.sqrt(np.mean((pred - yl) ** 2)))
        sse = float(np.sum((pred - yl) ** 2))
        r2 = float(1 - sse / sst_train)                                   # 正文口径：训练折均值
        r2_pooled = float(1 - sse / np.sum((yl - yl.mean()) ** 2))        # 旧口径：全样本均值（仅存档对照）
        var_instr = float(np.nanmean(sd_within ** 2))
        rows_d.append({
            "模型配置": cname,
            "按果分组 5 折 CV 的 R²（DM）": r2,
            "（对照）以全样本均值为 SST 的 R²": r2_pooled,
            "RMSEP（%DM）": rmsep,
            "同果跨设备 预测SD 中位": float(np.nanmedian(sd_within)),
            "同果跨设备 预测极差 中位": float(np.nanmedian(spread)),
            "同果跨设备 预测极差 P90": float(np.nanpercentile(spread.dropna(), 90)),
            "换台仪器就会变的那部分 RMSEP（√var_instr）": float(np.sqrt(var_instr)),
            "**仪器间分量占 MSEP 比例**": var_instr / rmsep ** 2,
        })
    tab_d = pd.DataFrame(rows_d)
    r2 = float(tab_d["按果分组 5 折 CV 的 R²（DM）"].max())   # 表e 用强模型的 R²

    # ---------- 表e：由实测 R² 反演参考值可靠性下界 ----------
    # 若参考真值为单次局部测定 y = mu + e，任何只感知果水平信息的模型
    # 对 y 的 R² 上界即参考可靠性 ICC ⇒ 实测 R² 构成 ICC 的下界。
    tab_e = pd.DataFrame([
        {"数据集": "001 苹果（单面参考）", "实测 R²": 0.113,
         "⇒ 参考可靠性 ICC 下界": 0.113, "直接实测 ICC": 0.477,
         "结论": "下界远松于实测值 ⇒ 天花板**不是**绑定约束，采集配置才是"},
        {"数据集": "012 猕猴桃（单次参考）", "实测 R²": r2,
         "⇒ 参考可靠性 ICC 下界": r2, "直接实测 ICC": np.nan,
         "结论": "无参考重复，但实测 R² 已迫使其参考可靠性 ≥ 该值"},
    ])

    log.log(tab_a.to_string(index=False))
    log.log("")
    log.log(tab_c.to_string(index=False))
    log.log("")
    log.log(tab_d.to_string(index=False))

    out = write_script_workbook(__file__, {
        0: ("表a：设计结构与可辨识性", tab_a),
        "表b：设备固定效应": tab_b,
        "表c：谱层面方差分解": tab_c,
        "表d：预测层面仪器间离散度": tab_d,
        "表e：由实测R²反演参考可靠性下界": tab_e,
    })
    log.log(f"→ {out}")
    sys.stdout = log._stdout  # noqa: SLF001


if __name__ == "__main__":
    main()
