"""
91instrprobe_v18.py: v18 候选 C1 的三项决定性预检——跨仪器残差的结构归因

背景（为什么跑这个脚本）:
    选题评审一致把候选 C1（跨仪器差异 = 可辨识的函数型观测算子）
    的生死线定在同一点：先前的预检只排除了"波长平移是主因"，并未排除"样本 × 仪器交互
    是主因"。若残差随样本基质变化，"样本无关算子"这个研究对象根本不存在，C1 的三个定理
    全部悬空。本脚本给出该问题的直接判据。

三项检验:
    P1 波长扭曲归因 —— 逐样本"仿射(增益+基线)" vs "仿射 + 最优全局波长平移"，
       残差降低幅度 = 波长漂移这一项贡献的上界。
    P2 迁移落差量尺 —— PLS(10 成分) 源仪器训练 → 目标仪器测试，与同仪器 5 折 CV 比倍数。
    P3 样本依赖性检验 —— (a) 共享仿射 vs 逐样本仿射；(b) 逐样本仿射系数能否被样本化学
       成分预测；(c) 扣仿射后残差主成分得分能否被化学成分预测（置换检验 500 次）。

方法论证: 见 docs/0NMETHOD_LEDGER.md §M2 / §M3 / §M4。
[免检] scipy.io.loadmat 读取、numpy SVD、样条重采样的具体调用写法 —— 纯工程实现。

运行方式:
    cd <project root>
    python code/91instrprobe_v18.py --device cpu

输出文件:
    outputs/91instrprobe_v18.xlsx           — 三项检验结果（sheet: warp / transfer_gap / sample_dependence）
    logs/91instrprobe_v18_YY-MM-DD_HHMMSS.log — 运行日志
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import (  # noqa: E402
    get_device,
    get_logger,
    log_experiment_header,
    set_random_seed,
    write_script_workbook,
)

import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_ROOT = DATA_ROOT
CORN_MAT = f"{DATA_ROOT}/036_corn_nir_3instruments/corn.mat"
TABLET_MAT = f"{DATA_ROOT}/037_tablet_nir_shootout2002/nir_shootout_2002.mat"
SEED = 2004  # Pilot dev 种子；严禁出现在 Formal 的 5 个固定种子中
N_PERM = 500
N_PLS_COMP = 10


# ------------------------------------------------------------------ 数据读取
def load_corn() -> tuple[dict[str, np.ndarray], np.ndarray, list[str], np.ndarray]:
    """036 玉米：同一 80 个样本在 m5/mp5/mp6 三台仪器上各测一遍。"""
    m = loadmat(CORN_MAT, struct_as_record=False, squeeze_me=False)
    spec = {k: m[f"{k}spec"][0, 0].data.astype(float) for k in ("m5", "mp5", "mp6")}
    props = m["propvals"][0, 0].data.astype(float)
    wl = np.arange(1100.0, 2500.0, 2.0)[: spec["m5"].shape[1]]
    return spec, props, ["水分", "油", "蛋白", "淀粉"], wl


def load_tablet() -> tuple[dict[str, np.ndarray], np.ndarray, list[str], np.ndarray]:
    """037 药片：同批 655 片跨 2 台仪器。三个 split 纵向拼回全集（配对关系按行对齐）。"""
    m = loadmat(TABLET_MAT, struct_as_record=False, squeeze_me=False)
    a = np.vstack([m[k][0, 0].data.astype(float) for k in ("calibrate_1", "test_1", "validate_1")])
    b = np.vstack([m[k][0, 0].data.astype(float) for k in ("calibrate_2", "test_2", "validate_2")])
    y = np.vstack([m[k][0, 0].data.astype(float) for k in ("calibrate_Y", "test_Y", "validate_Y")])
    ok = (a.std(1) > 1e-9) & (b.std(1) > 1e-9) & np.isfinite(y).all(1)
    wl = np.linspace(600.0, 1898.0, a.shape[1])
    return {"instr1": a[ok], "instr2": b[ok]}, y[ok], ["有效成分", "片重", "硬度"], wl


# ------------------------------------------------------------------ P1 波长扭曲归因
def per_sample_affine(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐样本拟合 tgt ≈ a·src + b。返回 (归一化残差, 系数矩阵 n×2)。"""
    coef = np.array([np.polyfit(src[i], tgt[i], 1) for i in range(src.shape[0])])
    resid = np.array(
        [
            np.sqrt(np.mean((tgt[i] - (coef[i, 0] * src[i] + coef[i, 1])) ** 2)) / tgt[i].std()
            for i in range(src.shape[0])
        ]
    )
    return resid, coef


def best_shift_resid(src: np.ndarray, tgt: np.ndarray, wl: np.ndarray,
                     shifts: np.ndarray) -> np.ndarray:
    """在逐样本仿射之上再叠一个全局波长平移 δ，网格搜索最优。返回 n×2 的 (残差, δ)。"""
    out = []
    for i in range(src.shape[0]):
        cs = CubicSpline(wl, src[i])
        best = (np.inf, 0.0)
        for d in shifts:
            warped = cs(np.clip(wl + d, wl[0], wl[-1]))
            k, c = np.polyfit(warped, tgt[i], 1)
            e = np.sqrt(np.mean((tgt[i] - (k * warped + c)) ** 2)) / tgt[i].std()
            if e < best[0]:
                best = (e, float(d))
        out.append(best)
    return np.array(out)


# ------------------------------------------------------------------ P2 迁移落差量尺
def snv(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def first_deriv(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=1)


PREPROCS: dict[str, Any] = {
    "原始": lambda x: x,
    "SNV": snv,
    "一阶导+SNV": lambda x: snv(first_deriv(x)),
}


def transfer_gap(src: np.ndarray, tgt: np.ndarray, y: np.ndarray, j: int) -> tuple[float, float]:
    """返回 (同仪器 5 折 CV RMSE, 跨仪器 RMSE)。源仪器全量训练 → 目标仪器全量测试。"""
    pls = PLSRegression(n_components=N_PLS_COMP).fit(src, y[:, j])
    cross = float(np.sqrt(np.mean((pls.predict(tgt).ravel() - y[:, j]) ** 2)))
    sq = []
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(src):
        p = PLSRegression(n_components=N_PLS_COMP).fit(src[tr], y[tr, j])
        sq.append((p.predict(src[te]).ravel() - y[te, j]) ** 2)
    within = float(np.sqrt(np.mean(np.concatenate(sq))))
    return within, cross


# ------------------------------------------------------------------ P3 样本依赖性
def _r2_on_chem(score: np.ndarray, chem_z: np.ndarray) -> float:
    """把 score 对标准化后的化学成分做线性回归，返回 R²。"""
    z = (score - score.mean()) / (score.std() + 1e-12)
    design = np.c_[np.ones(len(z)), chem_z]
    beta, *_ = np.linalg.lstsq(design, z, rcond=None)
    pred = design @ beta
    return float(1.0 - ((z - pred) ** 2).sum() / ((z - z.mean()) ** 2).sum())


def sample_dependence(src: np.ndarray, tgt: np.ndarray, y: np.ndarray,
                      rng: np.random.Generator, n_pc: int = 5) -> dict[str, Any]:
    """三问：增益/基线是否样本依赖？系数能否被化学成分预测？残差主方向能否被化学成分预测？"""
    n = src.shape[0]
    a_s, b_s = np.polyfit(src.ravel(), tgt.ravel(), 1)
    resid_shared = np.sqrt(((tgt - (a_s * src + b_s)) ** 2).mean(1)) / tgt.std(1)
    resid_per, coef = per_sample_affine(src, tgt)

    chem_z = (y - y.mean(0)) / y.std(0)
    r2_gain = _r2_on_chem(coef[:, 0], chem_z)
    r2_base = _r2_on_chem(coef[:, 1], chem_z)

    resid_mat = np.array([tgt[i] - (coef[i, 0] * src[i] + coef[i, 1]) for i in range(n)])
    u, s, _ = np.linalg.svd(resid_mat - resid_mat.mean(0), full_matrices=False)
    evr = (s**2 / (s**2).sum())[:n_pc]
    scores = u[:, :n_pc] * s[:n_pc]
    r2_pc = [_r2_on_chem(scores[:, k], chem_z) for k in range(n_pc)]

    # PC1 的置换零分布：打乱样本与化学成分的配对，破坏真实对应关系
    null = np.empty(N_PERM)
    for t in range(N_PERM):
        null[t] = _r2_on_chem(scores[:, 0], chem_z[rng.permutation(n)])
    p_val = float((null >= r2_pc[0]).mean())

    return {
        "resid_shared_median": float(np.median(resid_shared)),
        "resid_persample_median": float(np.median(resid_per)),
        "persample_extra_drop_pct": float(100 * (1 - np.median(resid_per) / np.median(resid_shared))),
        "r2_gain_on_chem": r2_gain,
        "r2_baseline_on_chem": r2_base,
        "pc1_var_ratio": float(evr[0]),
        "pc_var_ratios": np.round(evr, 4).tolist(),
        "r2_pc_on_chem": np.round(r2_pc, 4).tolist(),
        "pc1_perm_p": p_val,
        "pc1_perm_null_mean": float(null.mean()),
    }


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = get_device(args.device)
    set_random_seed(SEED, device=str(device))
    rng = np.random.default_rng(SEED)

    logger = get_logger("91instrprobe_v18")
    log_experiment_header(
        logger,
        {
            "脚本": "91instrprobe_v18.py",
            "目的": "v18 候选 C1 的三项决定性预检（波长扭曲归因 / 迁移落差量尺 / 样本依赖性）",
            "数据": "036 玉米(80×700×3 仪器) + 037 药片(655×650×2 仪器)，只读引用 05data",
            "种子": f"{SEED} (Pilot dev)",
            "设备": str(device),
            "阶段": "pilot",
            "置换次数": N_PERM,
            "PLS 成分数": N_PLS_COMP,
        },
    )

    corn_spec, corn_y, corn_names, corn_wl = load_corn()
    tab_spec, tab_y, tab_names, tab_wl = load_tablet()
    pairs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]] = [
        (f"036 {s}→{t}", corn_spec[s], corn_spec[t], corn_y, corn_names, corn_wl)
        for s, t in (("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6"))
    ]
    pairs.append(("037 instr1→instr2", tab_spec["instr1"], tab_spec["instr2"],
                  tab_y, tab_names, tab_wl))

    warp_rows, gap_rows, dep_rows = [], [], []
    shifts = np.arange(-15.0, 15.5, 0.5)

    for tag, src, tgt, y, ynames, wl in pairs:
        logger.log("=" * 60)
        logger.log(f"处理 {tag}  (n={src.shape[0]}, p={src.shape[1]})")

        # P1：037 样本多，抽 200 个做平移网格搜索以控时；036 全量
        idx = np.arange(src.shape[0])
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        r_aff, _ = per_sample_affine(src[idx], tgt[idx])
        r_shift = best_shift_resid(src[idx], tgt[idx], wl, shifts)
        warp_rows.append({
            "迁移方向": tag, "n_used": len(idx),
            "仅仿射_归一化残差_中位": round(float(np.median(r_aff)), 5),
            "仿射+最优平移_中位": round(float(np.median(r_shift[:, 0])), 5),
            "残差降低_pct": round(float(100 * (1 - np.median(r_shift[:, 0]) / np.median(r_aff))), 2),
            "最优平移_中位_nm": round(float(np.median(r_shift[:, 1])), 3),
            "最优平移_IQR低_nm": round(float(np.percentile(r_shift[:, 1], 25)), 3),
            "最优平移_IQR高_nm": round(float(np.percentile(r_shift[:, 1], 75)), 3),
            "平移同号比例": round(float(max((r_shift[:, 1] > 0).mean(), (r_shift[:, 1] < 0).mean())), 3),
            "波长采样间隔_nm": round(float(wl[1] - wl[0]), 3),
        })
        logger.log(f"  P1 波长扭曲：残差降低 {warp_rows[-1]['残差降低_pct']:.2f}%，"
                   f"最优平移中位 {warp_rows[-1]['最优平移_中位_nm']:.2f} nm"
                   f"（采样间隔 {warp_rows[-1]['波长采样间隔_nm']:.1f} nm）")

        # P2 迁移落差
        for pname, fn in PREPROCS.items():
            xs, xt = fn(src), fn(tgt)
            for j, yn in enumerate(ynames):
                within, cross = transfer_gap(xs, xt, y, j)
                gap_rows.append({
                    "迁移方向": tag, "预处理": pname, "指标": yn,
                    "同仪器CV_RMSE": round(within, 5), "跨仪器_RMSE": round(cross, 5),
                    "倍数": round(cross / within, 3),
                })
        logger.log(f"  P2 迁移落差：一阶导+SNV 后倍数 = "
                   f"{[r['倍数'] for r in gap_rows[-len(ynames):]]}")

        # P3 样本依赖性
        dep = sample_dependence(src, tgt, y, rng)
        dep_rows.append({"迁移方向": tag, **dep})
        logger.log(f"  P3 样本依赖：逐样本比共享额外降 {dep['persample_extra_drop_pct']:.1f}%；"
                   f"PC1 占残差方差 {100 * dep['pc1_var_ratio']:.1f}%；"
                   f"PC1 得分被化学成分解释 R²={dep['r2_pc_on_chem'][0]:.3f}，"
                   f"置换 p={dep['pc1_perm_p']:.4f}")

    out = write_script_workbook(__file__, {
        0: ("warp", pd.DataFrame(warp_rows)),
        "transfer_gap": pd.DataFrame(gap_rows),
        "sample_dependence": pd.DataFrame(dep_rows),
    })
    logger.log(f"已写出 {out}")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
