# Publication figure assets — COLOR / GRAYSCALE_PRINT_SAFE (presentation-only)

DS instruction 2026-09-14: "Scientific manuscript is frozen. This is
presentation-only work." This note documents the figure-regeneration
mechanics only; it makes no scientific claim and changes no accepted
value.

## What changed and what did not

`COLOR/` and `GRAYSCALE_PRINT_SAFE/` each contain the same 10 figure
files as the existing `manuscript_figures/` (same filenames, same figure
IDs, same figure content in the scientific sense). `manuscript_figures/`
itself is untouched. Every number, mean, confidence interval, axis range,
and label plotted in the new assets is read directly from
`canonical_inputs/R0{10..14}/*.csv` (already in this package, already
hash-verified) plus one additional source file,
`scripts/source_data/dynamics_by_epoch_R013.csv` — see "New source file"
below. No experiment was run and no value was altered to produce these
assets; only the rendering (palette, markers, linestyles, hatching, font
sizes, and — for fig01 only — the schematic's layout) changed.

## Figure 1 specifically

Fig. 1 ("Causal reconstruction of the historical observation") is a
schematic, not a data plot, so "preserve exact R010→R014 causal sequence"
means the same six steps in the same order and with the same wording. The
original single-row layout could not accommodate larger, readable text
without the boxes overlapping at publication size, so it was redrawn as a
two-row zig-zag flow (top row left→right: historical → R010 → R011; then
down; bottom row right→left: R012 → R013 → R014) — same six steps, same
order, same text, same arrows, just laid out with enough room for larger
type.

## Palette and distinguishability scheme

- **COLOR**: the Okabe–Ito eight-color qualitative palette (Okabe & Ito,
  2002; widely recommended for colorblind-safe scientific figures,
  including by *Nature*'s own graphics guidance) — distinguishable under
  the common forms of color-vision deficiency.
- **GRAYSCALE_PRINT_SAFE**: a fixed sequence of eight visually distinct
  gray levels standing in for the same eight palette slots.
- Every data series (dose, method, condition, or seed) is assigned one
  consistent `(marker shape, linestyle)` pair, used identically in BOTH
  palette variants, so a reader can match a series between the COLOR and
  GRAYSCALE_PRINT_SAFE versions of a figure by shape/dash alone —
  distinction never depends on color alone in either variant.
- Bar charts (Figs. 2–4) additionally use one hatch pattern per bar in
  both variants (visible but subtle in COLOR, load-bearing for
  distinguishing bars in GRAYSCALE_PRINT_SAFE).
- Base font sizes were increased throughout (13pt body / 14pt titles vs.
  the previous implicit matplotlib defaults) for publication-scale
  readability.

## New source file: `scripts/source_data/dynamics_by_epoch_R013.csv`

Figures 7a/7b (R013 embedding-RMS and gradient-L2 trajectories) require
per-epoch dynamics values that are not part of this package's
`canonical_inputs/` (which holds only final per-seed summary tables). The
original `generate_manuscript_figures.py` did not regenerate fig07a/fig07b
from source either — its own docstring states plots for R011–R014 were
"copied from accepted packages... remain checksum-bound rather than
regenerated." For this regeneration, the original per-epoch dynamics
export was retrieved directly from the accepted, already-DS-reviewed R013
canonical package
(`architect_reports/R013_embedding_scale_response_and_dynamics/manifests/dynamics_by_epoch.csv`
on Drive) and included here so fig07a/fig07b can be regenerated from an
explicit, checked-in source rather than re-copied as opaque images. Before
use, its epoch-15 mean embedding RMS per dose was cross-checked against
the manuscript's own prose (`D_below` ≈0.042, `D_ctor` ≈0.96–0.96 by
epoch 15) — exact match.

## Reproducing these assets

```
cd paper_track/review_candidate_v1
python3 scripts/generate_manuscript_figures_dual.py
```

Reads `canonical_inputs/` and `scripts/source_data/dynamics_by_epoch_R013.csv`
(both relative to the package root) and writes into `COLOR/` and
`GRAYSCALE_PRINT_SAFE/` in place. See `FIGURE_MANIFEST.csv` for the exact
sha256/size of every one of the 20 files as committed.
