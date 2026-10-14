"""
92xferpilot_v18.py: v18 候选 C1 的建设性 pilot——无标签结构化算子 vs 经典校准迁移

背景（为什么跑这个脚本）:
    预投稿评审指出 C1 的设计缺陷：四条预注册 kill criterion 全是 RMSEP 口径，
    而 C1 自认 RMSEP 降幅不足以支撑 novelty —— 等于 pilot 根本不检验它的核心主张。
    C1 真正的卖点是「**同一个仅用光谱（不用任何化学标签）估出的算子，能同时改善全部化学终点**」。
    本脚本直接检验这一条，并把主交付物从"不可能性结果"翻转为"能不能打赢 PDS"的建设性问题。

协议（严格防泄漏）:
    样本按 ID 划分为 train T / test E；桥接样本 B ⊂ T（|B| = m）。
    主模型 PLS 只在 **源仪器的 T** 上训练（用标签）。
    迁移算子只在 **B 的配对光谱** 上估计（**不用任何标签**）。
    评估在 **目标仪器的 E** 上进行（B 与 E 不相交）。

对比方法:
    none  无迁移直接套用       DS    直接标准化（PCR 正则）
    PDS   分段直接标准化       affine 共享逐波长增益+基线
    ours  共享逐波长仿射 + 秩 r 低秩残差算子（= C1 的结构化算子最简实现）

核心判据（非 RMSEP 口径，回应该意见）:
    - 跨终点一致性: 同一算子在全部 4 个（036）/3 个（037）终点上是否**同时**改善
    - 算子稳定性 : 不同桥接抽样下算子的相关（bootstrap 稳定性）

方法论证: 见 docs/0NMETHOD_LEDGER.md §M3 / §M5 / §M6。

运行方式:
    cd <project root>
    python code/92xferpilot_v18.py --device cpu

输出文件:
    outputs/92xferpilot_v18.xlsx              — sheet: rmsep / consistency / stability
    logs/92xferpilot_v18_YY-MM-DD_HHMMSS.log  — 运行日志
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import diags, eye as sp_eye, bmat as sp_bmat
from scipy.sparse.linalg import spsolve
from sklearn.cross_decomposition import PLSRegression

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
SEED = 2004
N_REPEAT = 30          # 随机桥接划分重复次数
BRIDGE_SIZES = (5, 10, 20, 40)
N_PLS = 10
PDS_WIN = 5            # PDS 半窗宽（波长点）
PDS_COMP = 2
DS_COMP = 15           # DS 的 PCR 成分数（正则化，否则 p>n 必过拟合）
LOWRANK_R = 3          # ours 的低秩残差算子秩
LAMBDA_SMOOTH = (1e0, 1e1, 1e2, 1e3, 1e4)   # Sobolev 粗糙罚候选（桥接样本内 LOO 选）


# ------------------------------------------------------------------ 数据
def load_pairs() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    """返回 [(标签, 源谱, 目标谱, Y, 终点名)]，行严格按样本 ID 对齐。"""
    out = []
    m = loadmat(f"{DATA_ROOT}/036_corn_nir_3instruments/corn.mat",
                struct_as_record=False, squeeze_me=False)
    spec = {k: m[f"{k}spec"][0, 0].data.astype(float) for k in ("m5", "mp5", "mp6")}
    y = m["propvals"][0, 0].data.astype(float)
    names = ["水分", "油", "蛋白", "淀粉"]
    for s, t in (("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6")):
        out.append((f"036 {s}→{t}", spec[s], spec[t], y, names))

    m2 = loadmat(f"{DATA_ROOT}/037_tablet_nir_shootout2002/nir_shootout_2002.mat",
                 struct_as_record=False, squeeze_me=False)
    a = np.vstack([m2[k][0, 0].data.astype(float) for k in ("calibrate_1", "test_1", "validate_1")])
    b = np.vstack([m2[k][0, 0].data.astype(float) for k in ("calibrate_2", "test_2", "validate_2")])
    yy = np.vstack([m2[k][0, 0].data.astype(float) for k in ("calibrate_Y", "test_Y", "validate_Y")])
    ok = (a.std(1) > 1e-9) & (b.std(1) > 1e-9) & np.isfinite(yy).all(1)
    out.append(("037 instr1→instr2", a[ok], b[ok], yy[ok], ["有效成分", "片重", "硬度"]))
    return out


# ------------------------------------------------------------------ 迁移算子（全部只用光谱，不用标签）
def fit_ds(src_b: np.ndarray, tgt_b: np.ndarray) -> Any:
    """直接标准化：找 F 使 tgt @ F ≈ src。用 PCR 正则（p ≫ n 时必须）。"""
    mu_t, mu_s = tgt_b.mean(0), src_b.mean(0)
    tc, sc = tgt_b - mu_t, src_b - mu_s
    u, s, vt = np.linalg.svd(tc, full_matrices=False)
    k = min(DS_COMP, (s > 1e-10).sum())
    pinv = vt[:k].T @ np.diag(1.0 / s[:k]) @ u[:, :k].T
    f = pinv @ sc
    return lambda x: (x - mu_t) @ f + mu_s


def fit_pds(src_b: np.ndarray, tgt_b: np.ndarray) -> Any:
    """分段直接标准化：逐波长用目标仪器邻域窗口回归源仪器该点。"""
    p = src_b.shape[1]
    coefs: list[tuple[int, int, np.ndarray, float]] = []
    for j in range(p):
        lo, hi = max(0, j - PDS_WIN), min(p, j + PDS_WIN + 1)
        win = tgt_b[:, lo:hi]
        ncomp = int(min(PDS_COMP, win.shape[1], max(1, win.shape[0] - 1)))
        pls = PLSRegression(n_components=ncomp).fit(win, src_b[:, j])
        coefs.append((lo, hi, pls.coef_.ravel(), float(pls.intercept_.ravel()[0])
                      if np.ndim(pls.intercept_) else float(pls.intercept_)))

    def apply(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for j, (lo, hi, c, b0) in enumerate(coefs):
            out[:, j] = x[:, lo:hi] @ c + b0
        return out

    return apply


def _d2_penalty(p: int) -> Any:
    """二阶差分算子 D2 的 DᵀD，用作 Sobolev 粗糙罚（惩罚 a(λ)、b(λ) 沿波长的曲率）。"""
    d2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(p - 2, p), format="csr")
    return (d2.T @ d2).tocsr()


def fit_affine(src_b: np.ndarray, tgt_b: np.ndarray, lam: float = 0.0) -> Any:
    """共享逐波长仿射 src[:,j] ≈ a_j·tgt[:,j] + b_j（样本无关）。

    lam = 0 时退化为逐波长独立最小二乘（无正则，桥接样本少时必过拟合）；
    lam > 0 时对 a(λ)、b(λ) 加 Sobolev 二阶粗糙罚，使算子沿波长光滑——
    这才是候选 C1 真正提出的"函数型"算子，而非逐点独立拟合。
    """
    p = src_b.shape[1]
    if lam <= 0:
        a = np.empty(p)
        b = np.empty(p)
        for j in range(p):
            v = tgt_b[:, j]
            if v.std() < 1e-12:
                a[j], b[j] = 1.0, float(src_b[:, j].mean() - v.mean())
            else:
                a[j], b[j] = np.polyfit(v, src_b[:, j], 1)
        return (lambda x: x * a + b), (a, b)

    n = src_b.shape[0]
    saa = (tgt_b**2).sum(0)              # Σ_i t_ij²
    sab = tgt_b.sum(0)                   # Σ_i t_ij
    sya = (src_b * tgt_b).sum(0)         # Σ_i s_ij t_ij
    syb = src_b.sum(0)                   # Σ_i s_ij
    pen = _d2_penalty(p) * (lam * n)
    top = sp_bmat([[diags(saa) + pen, diags(sab)]], format="csr")
    bot = sp_bmat([[diags(sab), diags(np.full(p, float(n))) + pen]], format="csr")
    mat = sp_bmat([[top], [bot]], format="csc")
    sol = spsolve(mat, np.concatenate([sya, syb]))
    a, b = sol[:p], sol[p:]
    return (lambda x: x * a + b), (a, b)


def _pick_lambda(src_b: np.ndarray, tgt_b: np.ndarray) -> float:
    """在桥接样本内用留一交叉验证选粗糙罚强度（只用光谱，不碰标签、不碰测试集）。"""
    n = src_b.shape[0]
    if n < 3:
        return LAMBDA_SMOOTH[-1]
    best, best_lam = np.inf, LAMBDA_SMOOTH[-1]
    for lam in LAMBDA_SMOOTH:
        err = 0.0
        for i in range(n):
            keep = np.arange(n) != i
            f, _ = fit_affine(src_b[keep], tgt_b[keep], lam=lam)
            err += float(np.mean((src_b[i] - f(tgt_b[i : i + 1])[0]) ** 2))
        if err < best:
            best, best_lam = err, lam
    return best_lam


def fit_ours(src_b: np.ndarray, tgt_b: np.ndarray, rank: int = LOWRANK_R,
             lam: float | None = None) -> Any:
    """C1 的结构化算子：**Sobolev 平滑的**共享逐波长仿射 + 秩 r 低秩残差算子。"""
    if lam is None:
        lam = _pick_lambda(src_b, tgt_b)
    aff, (a, b) = fit_affine(src_b, tgt_b, lam=lam)
    resid = src_b - aff(tgt_b)                      # 仿射之后剩下的部分
    tc = tgt_b - tgt_b.mean(0)
    u, s, vt = np.linalg.svd(tc, full_matrices=False)
    k = min(rank, (s > 1e-10).sum())
    if k == 0:
        return (lambda x: aff(x)), (a, b, np.zeros((tgt_b.shape[1], tgt_b.shape[1])))
    scores = u[:, :k] * s[:k]                       # 目标谱在前 k 个主方向上的得分
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(scores)), scores], resid, rcond=None)
    mu_t = tgt_b.mean(0)
    vk = vt[:k]

    def apply(x: np.ndarray) -> np.ndarray:
        sc = (x - mu_t) @ vk.T
        return aff(x) + np.c_[np.ones(len(x)), sc] @ coef

    return apply, (a, b, coef)


METHODS = ("none", "affine", "affine_smooth", "DS", "PDS", "ours")


# ------------------------------------------------------------------ 主流程
def run_pair(tag: str, src: np.ndarray, tgt: np.ndarray, y: np.ndarray,
             ynames: list[str], rng: np.random.Generator, logger: Any
             ) -> tuple[list[dict], list[dict], list[dict]]:
    n = src.shape[0]
    rmsep_rows: list[dict] = []
    stab_rows: list[dict] = []
    n_train = int(round(0.6 * n))

    for m_bridge in BRIDGE_SIZES:
        if m_bridge > n_train - 5:
            continue
        acc: dict[tuple[str, str], list[float]] = {(mth, yn): [] for mth in METHODS for yn in ynames}
        op_vectors: list[np.ndarray] = []

        for _ in range(N_REPEAT):
            perm = rng.permutation(n)
            tr, te = perm[:n_train], perm[n_train:]
            bridge = tr[:m_bridge]

            transforms: dict[str, Any] = {"none": (lambda x: x)}
            transforms["DS"] = fit_ds(src[bridge], tgt[bridge])
            transforms["PDS"] = fit_pds(src[bridge], tgt[bridge])
            aff_raw, _ = fit_affine(src[bridge], tgt[bridge], lam=0.0)
            transforms["affine"] = aff_raw          # 无正则逐点仿射（对照组）
            lam = _pick_lambda(src[bridge], tgt[bridge])
            aff_sm, (a_vec, _b_vec) = fit_affine(src[bridge], tgt[bridge], lam=lam)
            transforms["affine_smooth"] = aff_sm    # 仅平滑仿射，无低秩项（消融）
            ours, _ = fit_ours(src[bridge], tgt[bridge], lam=lam)
            transforms["ours"] = ours
            op_vectors.append(a_vec)                # 稳定性看**平滑后**的增益向量

            for j, yn in enumerate(ynames):
                pls = PLSRegression(n_components=N_PLS).fit(src[tr], y[tr, j])
                for mth in METHODS:
                    xt = transforms[mth](tgt[te])
                    pred = pls.predict(xt).ravel()
                    rmse = float(np.sqrt(np.mean((pred - y[te, j]) ** 2)))
                    acc[(mth, yn)].append(rmse / (y[:, j].std() + 1e-12))  # 标准化 RMSEP

        for mth in METHODS:
            for yn in ynames:
                v = np.array(acc[(mth, yn)])
                rmsep_rows.append({
                    "迁移方向": tag, "桥接样本数": m_bridge, "方法": mth, "终点": yn,
                    "标准化RMSEP_均值": round(float(v.mean()), 5),
                    "标准化RMSEP_中位": round(float(np.median(v)), 5),
                    "标准化RMSEP_标准差": round(float(v.std(ddof=1)), 5),
                })

        # 算子稳定性：不同桥接抽样下逐波长增益向量之间的两两相关
        ops = np.array(op_vectors)
        cors = []
        for i in range(len(ops)):
            for k in range(i + 1, len(ops)):
                c = np.corrcoef(ops[i], ops[k])[0, 1]
                if np.isfinite(c):
                    cors.append(c)
        stab_rows.append({
            "迁移方向": tag, "桥接样本数": m_bridge,
            "增益向量两两相关_中位": round(float(np.median(cors)), 4) if cors else np.nan,
            "增益向量两两相关_P10": round(float(np.percentile(cors, 10)), 4) if cors else np.nan,
            "n_pairs": len(cors),
        })
        logger.log(f"  [{tag}] m={m_bridge}: 算子稳定性(增益向量相关中位)="
                   f"{stab_rows[-1]['增益向量两两相关_中位']}")

    # 跨终点一致性：同一算子是否在全部终点上同时优于最佳传统方法
    df = pd.DataFrame(rmsep_rows)
    cons_rows: list[dict] = []
    for m_bridge in sorted(df["桥接样本数"].unique()):
        sub = df[df["桥接样本数"] == m_bridge]
        piv = sub.pivot_table(index="终点", columns="方法", values="标准化RMSEP_均值")
        classic = piv[["DS", "PDS", "affine"]].min(axis=1)   # 传统方法组（不含本方法的消融项）
        win = (piv["ours"] < classic)
        rel = (classic - piv["ours"]) / classic
        cons_rows.append({
            "迁移方向": tag, "桥接样本数": int(m_bridge),
            "ours胜过最佳传统的终点数": int(win.sum()), "终点总数": int(len(win)),
            "全终点同时改善": bool(win.all()),
            "相对改善_中位_pct": round(float(100 * rel.median()), 2),
            "相对改善_最小_pct": round(float(100 * rel.min()), 2),
            "ours_vs_none_中位_pct": round(float(
                100 * ((piv["none"] - piv["ours"]) / piv["none"]).median()), 2),
        })
        logger.log(f"  [{tag}] m={m_bridge}: ours 胜过最佳传统 "
                   f"{cons_rows[-1]['ours胜过最佳传统的终点数']}/{cons_rows[-1]['终点总数']} 个终点，"
                   f"相对改善中位 {cons_rows[-1]['相对改善_中位_pct']}%")
    return rmsep_rows, cons_rows, stab_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = get_device(args.device)
    set_random_seed(SEED, device=str(device))
    rng = np.random.default_rng(SEED)

    logger = get_logger("92xferpilot_v18")
    log_experiment_header(logger, {
        "脚本": "92xferpilot_v18.py",
        "目的": "C1 建设性 pilot：无标签结构化算子 vs DS/PDS/affine/无迁移",
        "协议": "样本ID划分 T/E；桥接 B⊂T 只用光谱不用标签；PLS 主模型只在源仪器 T 上训练；评估在目标仪器 E",
        "重复": N_REPEAT, "桥接规模": str(BRIDGE_SIZES), "种子": f"{SEED} (Pilot dev)",
        "设备": str(device), "阶段": "pilot",
    })

    all_r, all_c, all_s = [], [], []
    for tag, src, tgt, y, ynames in load_pairs():
        logger.log("=" * 64)
        logger.log(f"处理 {tag}  n={src.shape[0]}  p={src.shape[1]}  终点={ynames}")
        r, c, s = run_pair(tag, src, tgt, y, ynames, rng, logger)
        all_r += r
        all_c += c
        all_s += s

    out = write_script_workbook(__file__, {
        0: ("rmsep", pd.DataFrame(all_r)),
        "consistency": pd.DataFrame(all_c),
        "stability": pd.DataFrame(all_s),
    })
    logger.log(f"已写出 {out}")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
