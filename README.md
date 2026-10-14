# Reference-value uncertainty in near-infrared fruit calibration

Code, intermediate result workbooks and the pre-registered statistical protocol for the
manuscript:

> **Within-fruit variability of reference soluble solids sets a ceiling on near-infrared
> calibration performance in apple**
> Panlin Li, Yuchang Li, Yutong Feng, Longjie Li, Ya Liu, Hua Huang\*
> Xinjiang Agricultural University · *manuscript in preparation*

---

## What this repository is for

Calibrations that predict fruit soluble solids content (SSC) or dry matter (DM) from
near-infrared spectra are judged by RMSEP and $R^2$ against a destructive reference
determination. That comparison carries a premise which is almost never examined: **that the
reference value is itself error-free.** It is not.

This repository lets you check every statistic in the paper against the released workbooks
without the raw data (except those of the cross-architecture replay in Supplementary S2.6,
whose intermediate products were not retained, so that replay can be repeated from the
scripts and seeds but not rechecked quantity by quantity), rerun the kiwifruit-side analyses
listed under *Reproducing* from the public data, and — more usefully — run the same
diagnostics on **your own** data:

| Deliverable | Where | What it answers |
|---|---|---|
| **Replicate design rule** | `code/93sscceiling_v18.py` | How many approximately independent spatial subsamples per specimen does a target $R^2$ require? |
| **Attenuation-ratio diagnostic** | `code/A4formal_v18.py`, `code/A5aggregate_v18.py` | Is your reported performance actually bound by reference-value variability? (parameter-free, falsifiable) |
| **Two-level grouped validation + audit** | `code/95splitleak_v18.py`, `code/A6formal_v18.py`, `code/A7datepos_v18.py` | Is your grouping level right — and is your "leakage" real, or an artefact of a target derived from the grouping variable? (the withdrawn day-level reading and its control test are §S4 and Fig. S1 of `paper/Supplementary_en.pdf`) |
| **Semi-synthetic degradation test** | `code/A9semisynth_v18.py`, `code/A10margin_v18.py` | Does reference noise degrade performance the way the theory predicts? (pre-registered) |

---

## Data availability

**The raw datasets are not redistributed here.** Three are used, with different statuses:

| Dataset | Role in the paper | Status |
|---|---|---|
| **Apple, 633 fruit × 5 destructive faces** (Xinjiang Agricultural University, 2025, three origins) | Reference-side variance components, attainable ceiling, design rule, attenuation diagnostic | **Internal, not previously published.** Not redistributed here: the holding institution's data policy does not permit public release. They may be requested from the corresponding author by email, and are shared at the institution's discretion. Every quantity derived from them is in this repository. |
| **Kiwifruit NIR dry matter / SSC** (5,418 fruit and 11,982 spectra; 4,318 fruit scanned on 2–5 devices) | Instrument-side error budget, semi-synthetic degradation test | Public dataset (figshare, doi:10.6084/m9.figshare.21747983.v1) — obtain it from its original source and place it under `$HSI_DATA_ROOT/012_kiwifruit_nir_drymatter/`. |
| **DeepHS-Fruit time-series hyperspectral** (2023 five-species release) | Validation-protocol audit: the withdrawn acquisition-day reading and its controls (Supplementary S4) | Public dataset — obtain it from <https://github.com/cogsys-tuebingen/deephs_fruit> and place it under `$HSI_DATA_ROOT/003_deephs_fruit_hyperspectral/DeepHS-Fruit-2023-Datasets/`. |

What **is** released here: all analysis code (apart from the two pieces listed under
*Layout*), every intermediate result workbook (`outputs/*.xlsx`), the raw per-run records
(`outputs/*_raw.json`: the Formal apple main table and leakage controls, the Formal
supplementary runs, the terminal-date decomposition, the semi-synthetic experiment and the
fixed-predictor comparison behind the retraining term),
all figures, and the
pre-registered statistical protocol. `docs/claim_evidence_map.md` binds each claim to the
script and workbook table it comes from; a few derived values that are closed-form functions
of stored components (for example the design table of Section 3.5) are recomputed rather than
stored, and dataset metadata are traceable to the data cards in `docs/` instead.

Process-level intermediates (e.g. the $B=2000$ bootstrap resamples) are not shipped; they
are replayable from the scripts and the fixed seeds.

---

## Reproducing

```bash
pip install -r requirements.txt
export HSI_DATA_ROOT=/path/to/your/data   # 012_kiwifruit_nir_drymatter/ (+ 003_deephs_fruit_hyperspectral/ for the time-series controls)
python code/A3prep.py                     # kiwifruit csv -> 03data/processed/v18_kiwi_instr.parquet
python code/97instrbudget_v18.py          # instrument-side error budget
python code/A9semisynth_v18.py            # semi-synthetic degradation (pre-registered V9b)
python code/A10margin_v18.py              # margin rule inversion + lookup
python code/B0figures_v18.py              # all figures
python code/B1graphabs.py                 # graphical abstract (HTML with inline SVG -> PDF via headless Chrome)
```

The time-series controls start from `code/98deephstraj_v18.py`, which writes
`03data/processed/v18_deephs_traj.parquet` from the DeepHS-Fruit release; the Formal
day-level runs and the controls with non-derived targets (Supplementary S4) sit inside
`A6formal` and `A4formal`, which also need the apple data.

Scripts that need the apple data (`93sscceiling`, `94localsense`, `95splitleak`,
`96satdiag`, `A1consolidate`, `A2forensicfix`, `A4formal`, `A6formal`, `A8meta`,
`A11derived`, `A15maskcheck`) will not run, or not in full, without it; their **outputs are
included** in `outputs/` so the downstream analyses and every number in the paper, except
those of the cross-architecture replay, remain checkable.

**Reproducibility.** All models are PLS and closed-form moment estimators — no tensor
operations, CPU only (`torch` is in `requirements.txt` only because `code/export_utils.py`
imports it at module level; the CPU build is enough). Given the seed, a rerun on the same
architecture and BLAS implementation returns the reported values bit for bit; across
architectures the selected component count can move (Supplementary S2.6). Everything split-dependent is run over
five seeds fixed in advance: `[20060515, 20041210, 19810915, 2023, 2024]`. Confidence
intervals use a two-level cluster bootstrap with **seeds as clusters** (an i.i.d. bootstrap
over the 100 repeated splits underestimates the width, because the 20 repeats under one seed
share a random stream).

---

## Layout

```
code/       25 scripts + export_utils.py
outputs/    22 result workbooks (.xlsx) + 5 files of raw per-run records (.json)
figures/    the 7 figures of the manuscript, vector PDF
docs/       pre-registered statistical protocol, method ledger, figure audit,
            claim–evidence map, dataset datasheets
paper/      the manuscript and the Supplementary Material (PDF)
```

A figure file carries the name the script writes it under, which is not the number it
carries in the manuscript — one main figure moved into the Supplementary Material
during revision:

| In the manuscript | File in `figures/` |
|---|---|
| Figure 1 | `B0figures_v18-fig-a.pdf` |
| Figure 2 | `B0figures_v18-fig-b.pdf` |
| Figure 3 | `B0figures_v18-fig-c.pdf` |
| Figure 4 | `B0figures_v18-fig-e.pdf` |
| Figure S1 | `B0suppfig_v18-fig-a.pdf` |
| Figure S2 | `B0figures_v18-fig-d.pdf` |
| Graphical abstract | `B1graphabs.pdf` |

`code/91instrprobe_v18.py` and `code/92xferpilot_v18.py` are idea-stage pre-checks on two
other public NIR datasets (corn on three instruments; the 2002 pharmaceutical-tablet
shootout); no number from them enters the paper and their outputs are not included.

Two pieces that `docs/` refers to are not in `code/`: the literature-metadata harvester
(it produces no number that enters the paper) and the apple half of the preprocessing step,
`A3prep_server_v18.py::prep_apple` (it only runs against the unpublished raw apple data; the
kiwifruit half is `code/A3prep.py`).

Run logs and model checkpoints are runtime products: the scripts create `logs/` and
`checkpoints/` on first run, and neither is shipped with this repository.

### Notes on `docs/`

- `0NSTATISTICAL_PROTOCOL.md` — the analysis plan, cleaning rules and every pass criterion,
  **written down and archived before the final results were seen**. Appendix G records a
  pre-registration whose premise turned out to be wrong (V9) and is **kept verbatim rather
  than deleted**; Appendix H is its corrected successor (V9b) with no threshold changed.
- `claim_evidence_map.md` — the claim-by-claim map: each claim bound to the script and
  workbook table it comes from (for the cross-architecture replay, the table that records
  each replayed value; the replay's intermediate products were not retained). The numbers
  were checked against these workbooks in two independent zero-context audits; both returned
  FAIL on the first pass, and what they found and how each item was fixed is recorded here.

---

## Citing

Please cite the manuscript. This repository is archived as its reproducibility material.

## License

Code: MIT (see `LICENSE`). The result workbooks and figures are released under
CC BY 4.0. The raw datasets are **not** covered by either — see *Data availability* above.
