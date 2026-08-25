#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A8meta_v18 — 把稿件里的「数据集元信息」类断言实测落盘，补齐可采信证据。

起因（claim audit R6 · findings 2/3）：稿件正文里有一批**元信息**断言在工作簿里查不到出处——
猕猴桃波长范围 402-1065 nm、苹果每果 5 面的采样几何、633 果/3149 行、姊妹稿 015 所用的
2025 三产地子集样本量（山东 179 / 新疆 166 / 甘肃 225）、DeepHS 采集日与成像日序一一对应。
这些数字本身没错，但审计只认工作簿、不认 `.md` 数据卡里的自述，因而全部记为 missing_evidence。

本脚本**不做任何建模**，只把这些量从产物 parquet（以及 015 的公开产物 npz）**实测**出来，
逐条带上「从哪个文件、怎么算的」写进工作簿，使其可被零上下文审计核验。

关于「苹果 5 面采样几何」：数据本身只能证明**每果恰好 5 个具名面**（顶部/底部/侧面1-3），
**不能证明侧面之间相隔 120°**——后者是采集方案的记录，属外部元信息。本脚本如实区分这两者，
不把"5 个具名侧面"冒充成"实测 120°"。

[免检] 纯元信息统计（计数、取极值、唯一性核验），不产生任何进入结论的推断量。

运行方式:
    python3 code/A8meta_v18.py

输出文件:
    outputs/A8meta_v18.xlsx  — 表a 元信息实测；表b 外部元信息（数据本身无法证明者）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger, write_script_workbook  # noqa: E402

PROC = "03data/processed"
import os

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
# 原始数据集见 README 的「数据可用性」一节；001/002 为新疆农业大学内部数据，未随本仓库发布。
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
D015 = os.path.join(os.environ.get("SIBLING_015_DIR", "../015"),
                    "03data", "processed", "benchmarks", "apple.npz")

logger = Logger(__file__)


def main() -> None:
    rows: list[dict] = []

    def add(claim, value, how, src):
        rows.append({"稿件断言": claim, "实测值": value, "计算方式": how, "证据来源文件": src})

    # ── 苹果 ──────────────────────────────────────────────────────────────
    ap = pd.read_parquet(f"{PROC}/v18_apple_faces.parquet")
    wl = np.array([float(c[:-2]) for c in ap.columns if c.endswith("nm")])
    faces = sorted(ap["face"].unique())
    per_fruit = ap.groupby("id")["face"].nunique()
    add("苹果清洗后 633 果", int(ap["id"].nunique()), "id 唯一值计数",
        "v18_apple_faces.parquet")
    add("配对成功 3,149 行（谱×面）", int(len(ap)), "行数", "v18_apple_faces.parquet")
    add("3,165 次参考测定 = 633 果 × 5 面", int(ap["id"].nunique() * 5),
        "果数 × 5（参考测定在配对前完成，故 >3,149 配对行）", "v18_apple_faces.parquet")
    # ⚠️ 「每果恰好 5 面、设计平衡」是对**参考值表**（方差分量分析的输入）而言，
    # 不是对光谱配对表而言。二者必须分开核，否则会把配对损失误判成设计不平衡。
    ssc = pd.read_csv(os.path.join(
        os.path.join(DATA_ROOT, "001_apple_hyperspectral_multiyear"),
        "新疆-山东-甘肃苹果糖度数据2025.csv"), encoding="gbk")
    ssc.columns = ["id", "顶部", "底部", "侧面1", "侧面2", "侧面3"]
    rawv = ssc[["顶部", "底部", "侧面1", "侧面2", "侧面3"]].apply(pd.to_numeric, errors="coerce")
    kept = rawv.drop(index=rawv[(rawv > 20.0).any(axis=1)].index).dropna()
    add("方差分量分析：每果恰好 5 面、设计平衡（矩估计前提）",
        f"参考值表清洗后 {kept.shape[0]} 果 × {kept.shape[1]} 面 = {kept.size} 次测定；"
        f"逐果非缺失面数全为 5 = {bool(kept.notna().sum(1).eq(5).all())}",
        "读糖度 CSV，剔含 >20 °Brix 的整果后逐果计非缺失面数",
        "05data/001_.../新疆-山东-甘肃苹果糖度数据2025.csv")
    cnt = per_fruit.value_counts().sort_index().to_dict()
    add("光谱配对表：面数分布（**非**完全平衡，与上一行是两张表）",
        f"面名={faces}；逐果面数分布={cnt}；配对损失 {kept.size - len(ap)} 行",
        "逐果 face nunique 的分布；损失 = 3165 − 配对行数",
        "v18_apple_faces.parquet")
    add("苹果光谱量程止于 1001 nm", f"{wl.min():.2f}–{wl.max():.2f} nm，{len(wl)} 波段",
        "波长列名取 min/max/计数", "v18_apple_faces.parquet")

    # ── 猕猴桃 ────────────────────────────────────────────────────────────
    kw = pd.read_parquet(f"{PROC}/v18_kiwi_instr.parquet")
    kwl = np.array([float(c[1:]) for c in kw.columns if c.startswith("X")])
    n_dev = kw.groupby("sample_id")["device"].nunique()
    add("猕猴桃波长 402–1065 nm（5 台设备公共可用区间）",
        f"{kwl.min():.0f}–{kwl.max():.0f} nm，{len(kwl)} 波段",
        "波长列名取 min/max/计数", "v18_kiwi_instr.parquet")
    add("5,418 果 / 11,982 条光谱", f"{kw['sample_id'].nunique()} 果 / {len(kw)} 谱",
        "sample_id 唯一值计数 / 行数", "v18_kiwi_instr.parquet")
    add("4,318 果由 2–5 台设备重复扫描", int((n_dev >= 2).sum()),
        "逐果统计 device 唯一值数，取 ≥2 者", "v18_kiwi_instr.parquet")
    add("5 台全扫者仅 90 个", int((n_dev == 5).sum()), "同上，取 ==5 者",
        "v18_kiwi_instr.parquet")
    add("另有 1,100 果仅 1 台设备扫描", int((n_dev == 1).sum()), "同上，取 ==1 者",
        "v18_kiwi_instr.parquet")
    add("设备台数 5", int(kw["device"].nunique()), "device 唯一值计数",
        "v18_kiwi_instr.parquet")

    # ── DeepHS 时序 ───────────────────────────────────────────────────────
    tj = pd.read_parquet(f"{PROC}/v18_deephs_traj.parquet")
    ki = tj[tj["fruit"] == "Kiwi"]
    one2one = bool(ki.groupby("day")["day_idx"].nunique().eq(1).all()
                   and ki.groupby("day_idx")["day"].nunique().eq(1).all())
    add("时序高光谱 3,405 条 VIS 光谱 / 5 个物种", f"{len(tj)} 条 / {tj['fruit'].nunique()} 物种",
        "行数 / fruit 唯一值计数", "v18_deephs_traj.parquet")
    add("猕猴桃 27 个采集日与 27 个日序值一一对应",
        f"采集日 {ki['day'].nunique()} 个，day_idx {ki['day_idx'].nunique()} 个，"
        f"双向一一对应={one2one}",
        "day↔day_idx 双向 groupby nunique 全为 1", "v18_deephs_traj.parquet")
    add("day_idx 的最小/最大值（A7 极端日定义）",
        f"{ki['day_idx'].min():.0f} / {ki['day_idx'].max():.0f}",
        "min/max（该定义先于结果，不依赖 R²）", "v18_deephs_traj.parquet")
    add("硬度带标注谱条数 n=694 / 储藏天数 n=717",
        f"firmness {int(tj['firmness'].notna().sum())} / "
        f"storage_days {int(tj['storage_days'].notna().sum())}",
        "非缺失计数", "v18_deephs_traj.parquet")

    # ── 姊妹稿 015 的数据重叠声明 ─────────────────────────────────────────
    if os.path.exists(D015):
        z = np.load(D015, allow_pickle=True)
        got = {}
        for tag, key in (("山东", "2025_山东_B"), ("新疆", "2025_新疆_B"),
                         ("甘肃", "2025_甘肃_B")):
            if f"X__{key}" in z:
                got[tag] = int(z[f"X__{key}"].shape[0])
        add("姊妹稿 015 所用 2025 三产地子集：山东 179 / 新疆 166 / 甘肃 225",
            " / ".join(f"{k} {v}" for k, v in got.items()),
            "读 015 产物 npz，取各域 X 的行数（只读，不修改）",
            "015 03data/processed/benchmarks/apple.npz")
        add("015 每果一条谱、一个整果标签（本文用的逐面结构 015 未使用）",
            f"该 npz 每域 X 形状 = (n果, {z['X__2025_山东_B'].shape[1]} 波段)，"
            f"Y 形状 = (n果, 1) —— 无逐面维度",
            "检查 X/Y 的维数：仅果×波段、果×1，不含面维",
            "015 03data/processed/benchmarks/apple.npz")
    else:
        add("姊妹稿 015 子集样本量", "无法核验（015 产物不可达）",
            "文件不存在", D015)

    tab_a = pd.DataFrame(rows)

    # ── 表b：数据本身无法证明的外部元信息（如实标注，不冒充实测）──────────
    tab_b = pd.DataFrame([
        {"稿件断言": "三个侧面沿赤道带每旋转 120° 取样",
         "为何数据无法证明": "产物只记录面名（侧面1/2/3），不含角度字段；120° 是采集方案的记录",
         "证据类型": "外部元信息（采集方案）", "指针": "03data/processed/v18_apple_faces-datasheet.md"},
        {"稿件断言": "新疆农业大学 2025 年自行采集、单批次",
         "为何数据无法证明": "产物不含采集日期与批次字段",
         "证据类型": "外部元信息（采集记录）", "指针": "03data/processed/v18_apple_faces-datasheet.md"},
        {"稿件断言": "数显折光仪标称精度 ±0.2 °Brix",
         "为何数据无法证明": "仪器标称值来自厂商规格书，不在数据内",
         "证据类型": "外部元信息（仪器规格）", "指针": "03data/processed/v18_apple_faces-datasheet.md"},
        {"稿件断言": "引言所引文献典型值 R²>0.85、RMSEP 0.5–1.0 °Brix",
         "为何数据无法证明": "属文献引用，不是本文实验产物",
         "证据类型": "文献引用（归 citation-audit 核验）", "指针": "ref3 / ref6 / ref7"},
    ])

    for _, r in tab_a.iterrows():
        logger.log(f"  {str(r['稿件断言'])[:44]:44s} → {r['实测值']}")
    logger.log(f"表b：{len(tab_b)} 条外部元信息，如实标注为非实测")

    out = write_script_workbook(__file__, {
        0: ("元信息实测", tab_a),
        1: ("外部元信息（非实测）", tab_b),
    })
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
