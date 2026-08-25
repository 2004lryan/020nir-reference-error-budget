"""A10margin_v18 —— 余量规则（干净基线打 1% 折扣）的反解与实测查表。

引入原因（claim audit R8，2026-07-25）
------------------------------------
稿件 §3.7 写了「按余量规则修正后 m_req 由 9 与 7 分别改为 11 与 8」，
但这两个数字**是在稿件里手算的，任何脚本都没算过、任何工作簿都没有它们**。
这与 EF-001（衰减比只硬编码在绘图脚本里）是同一类坑，违反规程④：
「任何写进论文的数字，必须由某个分析脚本计算并导出到工作簿」。

本脚本把该规则形式化并落盘。它**不重跑任何模型**——退化轨迹与 P4 反解已由
`A9semisynth_v18.py` 产出，本脚本只在其结果之上做闭式反解与查表，因此可在本机秒级完成。

余量规则（§3.7 事前处置方案）
-----------------------------
    R²_clean' = (1 - MARGIN) · R²_clean                     # MARGIN = 0.01
    m_req'    = ceil[ (σf²/σy²) · R²_t / (R²_clean' - R²_t) ]

注意分母用 σy² 而非 σa²：半合成设置下模型可感知整个干净标签（见 §M17 四段式）。
这与式(6) 同型，只是把 R²_clean 换成打折后的 R²_clean'。

诚实性约束（本脚本刻意实现的）
------------------------------
若 m_req' 不在实验网格 M_LEVELS = [1..10, 14, 20, ∞] 内，**不做任何插值**，
而是如实标注「不在网格内·无实测」，并给出左右相邻档位的实测值供夹逼。
插值出来的 R² 不是实测值，把它写进论文就是编造。
"""

from __future__ import annotations

import json
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import os as _os_
_OUTA = os.path.join(HERE, "..", "outputs")
OUT = _OUTA if _os_.path.isdir(_OUTA) else os.path.join(HERE, "..", "04outputs")

MARGIN = 0.01  # §3.7 余量规则：干净基线打 1% 折扣
# 与 A9semisynth_v18.P4_TARGETS 保持一致（事前锁定，附录 H.3）
P4_TARGETS = [("DM", 0.70, False), ("DM", 0.75, True), ("DM", 0.78, True),
              ("SSC", 0.70, False), ("SSC", 0.75, True), ("SSC", 0.80, True)]


def _fmt(v: object) -> str:
    return "—" if v is None else f"{float(v):.4f}"


def main() -> None:
    pre = pd.read_excel(os.path.join(OUT, "A9semisynth_v18.xlsx"),
                        sheet_name="预注册 V9b 判定")
    traj = pd.read_excel(os.path.join(OUT, "A9semisynth_v18.xlsx"),
                         sheet_name="退化轨迹")
    traj = traj[pd.to_numeric(traj["m（参考重复次数）"], errors="coerce").notna()].copy()
    traj["m"] = pd.to_numeric(traj["m（参考重复次数）"]).astype(int)
    grid = sorted(traj["m"].unique())

    with open(os.path.join(OUT, "A9semisynth_v18_raw.json"), encoding="utf-8") as f:
        m_levels = json.load(f)["m_levels"]

    rows = []
    for tgt, rt, blind in P4_TARGETS:
        p = pre[pre["目标"] == tgt].iloc[0]
        r2c, sy2, sf2 = float(p["干净基线 R²_clean"]), float(p["σy²"]), float(p["σf²(标定)"])
        g = traj[traj["目标"] == tgt].set_index("m")

        def req(clean: float) -> tuple[float, int]:
            raw = (sf2 / sy2) * rt / (clean - rt)
            return raw, int(math.ceil(raw))

        raw0, m0 = req(r2c)
        r2c_disc = (1.0 - MARGIN) * r2c
        raw1, m1 = req(r2c_disc)

        def lookup(m: int) -> tuple[str, object, object]:
            if m in g.index:
                r = g.loc[m]
                return ("在网格内", float(r["实测 R²（对带噪标签）"]), float(r["cluster CI95 下限"]))
            return ("不在网格内·无实测", None, None)

        state0, obs0, lo0 = lookup(m0)
        state1, obs1, lo1 = lookup(m1)

        lower = max([x for x in grid if x < m1], default=None)
        upper = min([x for x in grid if x > m1], default=None)
        brack = ""
        if state1.startswith("不在"):
            lo_s = (f"m={lower}: {float(g.loc[lower, '实测 R²（对带噪标签）']):.4f}"
                    if lower is not None else "—")
            up_s = (f"m={upper}: {float(g.loc[upper, '实测 R²（对带噪标签）']):.4f}"
                    if upper is not None else "—")
            brack = f"{lo_s} / {up_s}"

        rows.append({
            "目标": tgt, "R²_t（事前指定）": rt,
            "是否盲检": "盲检" if blind else "已见(不计入)",
            "原 m_req(未取整)": raw0, "原 m_req": m0,
            "原 m_req 处实测 R²": obs0, "原 m_req 是否达标": (
                "—" if obs0 is None else ("达标" if obs0 >= rt else "未达标")),
            f"折扣后 R²_clean（×{1-MARGIN:.2f}）": r2c_disc,
            "修正 m_req(未取整)": raw1, "**修正 m_req**": m1,
            "修正 m_req 是否在实验网格内": state1,
            "修正 m_req 处实测 R²": obs1,
            "修正 m_req 处 CI95 下限": lo1,
            "修正后是否达标（均值口径）": (
                "无实测·不判定" if obs1 is None else ("达标" if obs1 >= rt else "未达标")),
            "无实测时的相邻档夹逼": brack or "—",
        })

    df = pd.DataFrame(rows)
    blind = df[df["是否盲检"] == "盲检"]
    # 只有「余量规则真的改变了 m_req」的目标才构成对该规则的检验：修正前后 m_req 相同的
    # 目标在修正前就已达标，规则在它们身上什么也没做。此前 n_direct 直接数「最后达标的」，
    # 把这两个也算了进去，得出 3/4，与正文的「仅一例被实测直接验证」自相矛盾。
    # （一致性审计 HP-PHANTOM-RESULT，2026-07-26）
    changed = blind[blind["**修正 m_req**"] != blind["原 m_req"]]
    n_unchanged = len(blind) - len(changed)
    n_direct = int((changed["修正后是否达标（均值口径）"] == "达标").sum())
    n_nogrid = int((changed["修正 m_req 是否在实验网格内"].str.startswith("不在")).sum())

    meta = pd.DataFrame([{
        "余量幅度 MARGIN": MARGIN,
        "实验 m 网格": str(m_levels),
        "盲检目标数": len(blind),
        "**其中 m_req 被余量规则改变的（= 真正参与检验的）**": len(changed),
        "m_req 未改变·不参与检验": n_unchanged,
        "被实测直接验证": n_direct,
        "修正 m_req 落在网格外·只能夹逼": n_nogrid,
        "结论口径": (f"{len(blind)} 个盲检目标中，余量规则只改变了 {len(changed)} 个的 m_req，"
                     f"另 {n_unchanged} 个在修正前就已达标、不构成对该规则的检验。"
                     f"在真正参与检验的 {len(changed)} 个中，{n_direct} 个被实测直接验证，"
                     f"{n_nogrid} 个的修正 m_req 不在网格内、只能由相邻档夹逼。"),
        "反解式": "m_req = ceil[ (σf²/σy²) · R²_t / (R²_clean·(1−MARGIN) − R²_t) ]",
        "备注": "本表不做任何插值；网格外一律标注无实测。",
    }])

    path = os.path.join(OUT, "A10margin_v18.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="表a：余量规则反解与查表", index=False)
        meta.to_excel(w, sheet_name="表b：口径与结论", index=False)

    print(f"→ {path}")
    for r in rows:
        print(f"  {r['目标']:3s} R²_t={r['R²_t（事前指定）']:.2f} [{r['是否盲检']}] "
              f"m_req {r['原 m_req']} → 修正 {r['**修正 m_req**']} "
              f"({r['修正 m_req 是否在实验网格内']}) "
              f"实测 {_fmt(r['修正 m_req 处实测 R²'])} "
              f"→ {r['修正后是否达标（均值口径）']}  夹逼[{r['无实测时的相邻档夹逼']}]")
    print(f"\n  【汇总】盲检 {len(blind)} 个：m_req 被改变 {len(changed)}（未改变 {n_unchanged}，不参与检验）；"
          f"其中直接实测验证 {n_direct}，网格外只能夹逼 {n_nogrid}")


if __name__ == "__main__":
    main()
