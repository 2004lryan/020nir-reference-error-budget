# An error budget for near-infrared fruit calibration

Code, intermediate result workbooks and the pre-registered statistical protocol for the
manuscript:

> **An error budget for near-infrared fruit calibration: a reference-replicate design rule
> and a falsifiable attenuation diagnostic**
> Panlin Li, Yuchang Li, Yutong Feng, Longjie Li, Ya Liu, Hua Huang\*
> Xinjiang Agricultural University · *manuscript in preparation*

---

## What this repository is for

Calibrations that predict fruit soluble solids content (SSC) or dry matter (DM) from
near-infrared spectra are judged by RMSEP and $R^2$ against a destructive reference
determination. That comparison carries a premise which is almost never examined: **that the
reference value is itself error-free.** It is not.

This repository lets you reproduce every statistic in the paper, and — more usefully — run
the same diagnostics on **your own** data:

| Deliverable | Where | What it answers |
|---|---|---|
| **Replicate design rule** | `code/93sscceiling_v18.py` | How many approximately independent spatial subsamples per specimen does a target $R^2$ require? |
| **Attenuation-ratio diagnostic** | `code/A4formal_v18.py`, `code/A5aggregate_v18.py` | Is your reported performance actually bound by reference-value variability? (parameter-free, falsifiable) |
| **Two-level grouped validation + audit** | `code/95splitleak_v18.py`, `code/A6formal_v18.py`, `code/A7datepos_v18.py` | Is your grouping level right — and is your "leakage" real, or an artefact of a target derived from the grouping variable? (the withdrawn day-level reading and its control test are §S1 and Fig. S1 of `paper/Supplementary_en.pdf`) |
| **Semi-synthetic degradation test** | `code/A9semisynth_v18.py`, `code/A10margin_v18.py` | Does reference noise degrade performance the way the theory predicts? (pre-registered) |

---

## Data availability

**The raw datasets are not redistributed here.** Two are used, with different statuses:

| Dataset | Role in the paper | Status |
|---|---|---|
| **Apple, 633 fruit × 5 destructive faces** (Xinjiang Agricultural University, 2025, three origins) | Reference-side variance components, attainable ceiling, design rule, attenuation diagnostic | **Internal, not previously published.** Not redistributed here: the holding institution's data policy does not permit public release. They may be requested from the corresponding author by email, and are shared at the institution's discretion. Everything derived from them is in this repository. |
| **Kiwifruit NIR dry matter / SSC** (5,418 fruit, 11,982 spectra, 2–5 devices) | Instrument-side error budget, semi-synthetic degradation test | Public dataset — obtain it from its original source and place it under `$HSI_DATA_ROOT`. |

What **is** released here is everything derived from them: all analysis code, every
intermediate result workbook (`outputs/*.xlsx`), the raw per-run records of the
semi-synthetic experiment (`outputs/A9semisynth_v18_raw.json`), all figures, and the
pre-registered statistical protocol. **Every statistic in the manuscript is traceable to a
specific table in `outputs/`** — see `docs/claim_evidence_map.md` for the claim-by-claim map.

Process-level intermediates (e.g. the $B=2000$ bootstrap resamples) are not shipped; they
are replayable from the scripts and the fixed seeds.

---

## Reproducing

```bash
pip install -r requirements.txt
export HSI_DATA_ROOT=/path/to/your/data   # must contain 012_kiwifruit_nir_drymatter/
python code/97instrbudget_v18.py          # instrument-side error budget
python code/A9semisynth_v18.py            # semi-synthetic degradation (pre-registered V9b)
python code/A10margin_v18.py              # margin rule inversion + lookup
python code/B0figures_v18.py              # all figures
python code/B1graphabs.py                 # graphical abstract (FigureSpec -> SVG/PDF)
```

Scripts that need the apple data (`93sscceiling`, `94localsense`, `95splitleak`,
`96satdiag`, `A1consolidate`, `A2forensicfix`) will not run without it; their **outputs are
included** in `outputs/` so the downstream analyses and every number in the paper remain
checkable.

**Reproducibility.** All models are PLS and closed-form moment estimators — no tensor
operations, CPU only, deterministic given the seed. Everything split-dependent is run over
five seeds fixed in advance: `[20060515, 20041210, 19810915, 2023, 2024]`. Confidence
intervals use a two-level cluster bootstrap with **seeds as clusters** (an i.i.d. bootstrap
over the 100 repeated splits underestimates the width, because the 20 repeats under one seed
share a random stream).

---

## Layout

```
code/       21 analysis scripts + export_utils.py
outputs/    18 result workbooks (.xlsx) + 2 files of raw per-run records (.json)
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

Two scripts that `docs/` refers to are not in `code/`: the literature-metadata harvester
(it produces no number that enters the paper) and the raw-data preprocessing script
(it only runs against the unpublished raw apple data).

Run logs and model checkpoints are runtime products: the scripts create `logs/` and
`checkpoints/` on first run, and neither is shipped with this repository.

### Notes on `docs/`

- `0NSTATISTICAL_PROTOCOL.md` — the analysis plan, cleaning rules and every pass criterion,
  **written down and archived before the final results were seen**. Appendix G records a
  pre-registration whose premise turned out to be wrong (V9) and is **kept verbatim rather
  than deleted**; Appendix H is its corrected successor (V9b) with no threshold changed.
- `claim_evidence_map.md` — the claim-by-claim map: every number in the paper bound to the
  workbook table it comes from. Each number was checked against these workbooks in two
  independent zero-context audits; both returned FAIL on the first pass, and what they
  found and how each item was fixed is recorded here.

---

## Citing

Please cite the manuscript. This repository is archived as its reproducibility material.

## License

Code: MIT (see `LICENSE`). The result workbooks and figures are released under
CC BY 4.0. The raw datasets are **not** covered by either — see *Data availability* above.
