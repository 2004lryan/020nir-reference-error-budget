#!/usr/bin/env python3
"""A3prep —— 把公开的猕猴桃原始 CSV 收敛成分析脚本读取的预处理产物。

97instrbudget、A12opscan 与 A1consolidate 直接读原始 CSV；A6formal、A8meta、A9semisynth、A11derived、
A16retrainterm 读本脚本写出的 `03data/processed/v18_kiwi_instr.parquet`，重跑这几个脚本之前
须先跑本脚本：

    export HSI_DATA_ROOT=/path/to/your/data   # 含 012_kiwifruit_nir_drymatter/kiwifruit_dat.csv
    python code/A3prep.py

本文件是预处理步骤 A3prep_server_v18 的猕猴桃一段（prep_kiwi），数据处理部分逐行照录；原步骤
末尾另写数据卡，发布版的数据卡在 docs/，这里不再生成。清洗口径同
97instrbudget，不引入任何新的过滤或变换：dropna(sample_id, device)；剔除在全部 5 台设备上都缺
的 24 个波段（1068–1137 nm），取公共可用区间；波段列存为 float32。样本统计见
docs/v18_kiwi_instr-datasheet.md；数据卡里记的产物 SHA-256 会随 pandas/pyarrow 写入的版本
元数据而变，核对内容请比样本统计。

同一步骤的苹果一段（prep_apple）只对未公开的原始苹果数据运行，未随本仓发布；由苹果数据派生的
结果见 outputs/。

[免检] parquet + zstd 压缩 —— 纯传输格式选择，不影响任何数字结果。

| 修订日期 | 轮次 | 改了什么 | 为什么改 |
|---|---|---|---|
| 2026-10-14 | 1 | 随本仓发布 prep_kiwi 一段 | 上列五个脚本读它的产物，不发布就无法从公开数据重跑 |
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_utils import Logger

# 数据根目录：默认取环境变量 HSI_DATA_ROOT，未设时回退到 ../data
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
D012 = os.path.join(DATA_ROOT, "012_kiwifruit_nir_drymatter", "kiwifruit_dat.csv")
OUT = "03data/processed"

logger = Logger(__file__)


# ── 猕猴桃仪器侧 ─────────────────────────────────────────────────────
def prep_kiwi() -> None:
    df = pd.read_csv(D012, low_memory=False)
    wave = sorted([c for c in df.columns if c.startswith("X") and c[1:].isdigit()],
                  key=lambda c: int(c[1:]))
    df = df.dropna(subset=["sample_id", "device"])
    bad = [c for c in wave if df[c].isna().any()]   # 1068–1137 nm：5 台设备全缺
    wave = [c for c in wave if c not in bad]
    keep_cols = [c for c in ("sample_id", "device", "DM", "SSC") if c in df.columns] + wave
    sub = df[keep_cols].copy()
    for c in wave:
        sub[c] = sub[c].astype("float32")           # 传输体积减半，精度远超 PLS 需要

    out = os.path.join(OUT, "v18_kiwi_instr.parquet")
    sub.to_parquet(out, index=False, compression="zstd")
    n_dev = sub.groupby("sample_id")["device"].nunique()
    logger.log(f"[猕猴桃] {len(sub)} 谱 / {sub['sample_id'].nunique()} 果 / {len(wave)} 波段"
               f"（剔除全缺波段 {len(bad)} 个）→ {out}  {os.path.getsize(out)/1e6:.1f} MB")
    logger.log(f"  其中 {(n_dev >= 2).sum()} 个果由 ≥2 台设备扫描，{(n_dev == 5).sum()} 个果被 5 台全扫")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    prep_kiwi()


if __name__ == "__main__":
    main()
