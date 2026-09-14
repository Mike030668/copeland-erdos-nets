#!/usr/bin/env python3
"""Publication-quality dual COLOR / GRAYSCALE_PRINT_SAFE manuscript figures.

Presentation-only regeneration (DS instruction 2026-09-14): the scientific
manuscript is frozen; this script changes NOTHING about data, means, CIs,
axes, labels, figure IDs, or accepted lineage -- every number plotted here
is read verbatim from canonical_inputs/R0{10..14}/*.csv (the same files
already accepted and hash-verified in this package) plus
source_data/dynamics_by_epoch_R013.csv (the original R013 per-epoch
dynamics export, retrieved from the accepted R013 canonical package on
Drive -- it was never copied into canonical_inputs/ because the earlier
generate_manuscript_figures.py did not regenerate fig07a/fig07b from
source; this script does, from the same original numbers).

Produces, under COLOR/ and GRAYSCALE_PRINT_SAFE/ (identical filenames in
both), the same 10 image files as the existing manuscript_figures/:
  fig01_causal_reconstruction.png            (schematic, no data)
  fig02_R010_test_ppl.png
  fig03_R011_factorial_test_ppl.png
  fig04_R012_factorial_test_ppl.png
  fig05_R013_dose_response_test_ppl.png
  fig06_R014_dose_response_test_ppl.png
  fig07a_R013_embedding_rms_trajectory.png
  fig07b_R013_embedding_grad_l2_trajectory.png
  fig07c_R014_best_epoch_by_dose.png
  fig07d_R014_final_vs_best_val_ppl.png

Design: one consistent (marker, linestyle) per data series across BOTH
palette variants, so a reader can match a COLOR-figure series to its
GRAYSCALE_PRINT_SAFE counterpart by shape/dash alone. COLOR uses the
Okabe-Ito colorblind-safe qualitative palette. GRAYSCALE_PRINT_SAFE uses
distinct gray levels plus the same markers/linestyles, and hatching for
bar charts, so every distinction still works with color removed entirely.
"""
import csv
import math
import statistics as stats
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This script lives in scripts/ under the package root; canonical_inputs/,
# source_data/, and the COLOR/GRAYSCALE_PRINT_SAFE output directories are
# all siblings of scripts/, i.e. one level up.
ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "canonical_inputs"
SRC = ROOT / "source_data"
OUT_COLOR = ROOT / "COLOR"
OUT_GRAY = ROOT / "GRAYSCALE_PRINT_SAFE"
for d in (OUT_COLOR, OUT_GRAY):
    d.mkdir(parents=True, exist_ok=True)

# ---- shared style: larger, publication-readable typography ----
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
})

# Okabe-Ito colorblind-safe qualitative palette.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#F0E442", "#000000"]
# Distinct grayscale levels (dark -> light), paired 1:1 with the palette above by index.
GRAYS = ["0.05", "0.35", "0.55", "0.70", "0.85", "0.20", "0.45", "0.60"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 5, 1, 5))]
HATCHES = ["", "//", "xx", "..", "\\\\", "++", "oo", "--"]


def series_style(i, palette):
    color = OKABE_ITO[i % len(OKABE_ITO)] if palette == "color" else GRAYS[i % len(GRAYS)]
    return {
        "color": color,
        "marker": MARKERS[i % len(MARKERS)],
        "linestyle": LINESTYLES[i % len(LINESTYLES)],
        "hatch": HATCHES[i % len(HATCHES)],
    }


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def t_ci95(vals):
    n = len(vals)
    m = stats.mean(vals)
    if n < 2:
        return m, m, m
    sd = stats.stdev(vals)
    se = sd / math.sqrt(n)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}.get(n - 1, 1.96)
    h = tcrit * se
    return m, m - h, m + h


# =====================================================================
# fig01 — causal reconstruction schematic (no data; larger readable text)
# =====================================================================
def make_fig01(palette):
    steps = [
        "Historical\nselective-init gain",
        "R010\nattention-only attribution fails",
        "R011\ndominant contrast follows\ntoken embedding",
        "R012\ninitial RMS scale accounts for\ndominant factorial contrast",
        "R013\nfresh-seed\ndose response",
        "R014\nordering reproduces at one\nlarger tested capacity",
    ]
    # Two-row zig-zag layout (top row left->right, bottom row right->left) so
    # larger, readable text has room without boxes overlapping: 3 columns
    # instead of 6, at wide-enough per-column spacing for the longest box
    # text (R012's three-line label is the widest).
    COL = 1.65  # column spacing -- wide enough that the longest (R012) box's
                # text does not reach the neighboring column's arrow.
    positions = [(0, 1), (COL, 1), (2 * COL, 1), (2 * COL, 0), (COL, 0), (0, 0)]
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.axis("off")
    boxes = []
    for i, (text, (x, y)) in enumerate(zip(steps, positions)):
        st = series_style(i, palette)
        boxcolor = st["color"] if palette == "color" else "white"
        edgecolor = "black" if palette == "grayscale" else st["color"]
        ax.text(x, y, text, ha="center", va="center", fontsize=13, fontweight="bold",
                 linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.7", facecolor=boxcolor, edgecolor=edgecolor,
                           linewidth=2.0, alpha=0.35 if palette == "color" else 1.0),
                 color="black", zorder=3)
        boxes.append((x, y))
    # arrows: 0->1->2 (top, rightward), 2->3 (down), 3->4->5 (bottom, leftward)
    arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for a, b in arrow_pairs:
        xa, ya = boxes[a]; xb, yb = boxes[b]
        if ya == yb:
            dx = 0.62 if xb > xa else -0.62
            xytext = (xa + dx, ya); xy = (xb - dx, yb)
        else:
            xytext = (xa, ya - 0.22); xy = (xb, yb + 0.22)
        ax.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="-|>", lw=2.4, color="black"))
    ax.set_xlim(-0.95, 2 * COL + 0.95)
    ax.set_ylim(-0.55, 1.55)
    fig.tight_layout()
    return fig


# =====================================================================
# fig02 — R010 attention-only reconstruction (bar chart, method means)
# =====================================================================
def make_fig02(palette):
    rows = read_csv(CANON / "R010" / "per_seed.csv")
    order = ["xavier_g1.0", "orthogonal", "xavier_g1.2", "ce_lcg"]
    labels = ["xavier_g1.0", "orthogonal", "xavier_g1.2", "ce_lcg"]
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(float(r["test_ppl"]))
    means = [stats.mean(by_method[m]) for m in order]
    cis = [t_ci95(by_method[m]) for m in order]
    errs = [(m - lo, hi - m) for (m, lo, hi) in cis]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (label, mean) in enumerate(zip(labels, means)):
        st = series_style(i, palette)
        color = st["color"] if palette == "color" else "white"
        edgecolor = "black"
        ax.bar(i, mean, color=color, edgecolor=edgecolor, hatch=st["hatch"], linewidth=1.3,
               yerr=[[errs[i][0]], [errs[i][1]]], capsize=5,
               error_kw=dict(elinewidth=1.4, capthick=1.4))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Held-out test PPL")
    ax.set_title("R010 corrected attention-only reconstruction")
    fig.tight_layout()
    return fig


# =====================================================================
# fig03 — R011 embedding x attention factorial (cell means, bar chart)
# =====================================================================
def make_fig03(palette):
    rows = read_csv(CANON / "R011" / "per_seed.csv")
    order = ["A0B0", "A0B1", "A1B0", "A1B1"]
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(float(r["test_ppl"]))
    means = [stats.mean(by_cell[c]) for c in order]
    cis = [t_ci95(by_cell[c]) for c in order]
    errs = [(m - lo, hi - m) for (m, lo, hi) in cis]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (label, mean) in enumerate(zip(order, means)):
        st = series_style(i, palette)
        color = st["color"] if palette == "color" else "white"
        ax.bar(i, mean, color=color, edgecolor="black", hatch=st["hatch"], linewidth=1.3,
               yerr=[[errs[i][0]], [errs[i][1]]], capsize=5,
               error_kw=dict(elinewidth=1.4, capthick=1.4))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Held-out test PPL")
    ax.set_title("R011 embedding × attention factorial")
    fig.tight_layout()
    return fig


# =====================================================================
# fig04 — R012 scale x redraw factorial (cell means, bar chart)
# =====================================================================
def make_fig04(palette):
    rows = read_csv(CANON / "R012" / "per_seed.csv")
    order = ["S0D0", "S0D1", "S1D0", "S1D1"]
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(float(r["test_ppl"]))
    means = [stats.mean(by_cell[c]) for c in order]
    cis = [t_ci95(by_cell[c]) for c in order]
    errs = [(m - lo, hi - m) for (m, lo, hi) in cis]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (label, mean) in enumerate(zip(order, means)):
        st = series_style(i, palette)
        color = st["color"] if palette == "color" else "white"
        ax.bar(i, mean, color=color, edgecolor="black", hatch=st["hatch"], linewidth=1.3,
               yerr=[[errs[i][0]], [errs[i][1]]], capsize=5,
               error_kw=dict(elinewidth=1.4, capthick=1.4))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Held-out test PPL")
    ax.set_title("R012 scale × redraw decomposition")
    fig.tight_layout()
    return fig


# =====================================================================
# fig05 — R013 five-dose response (mean +/- CI, per-seed scatter)
# =====================================================================
def make_dose_response(palette, cycle, doses, title, filename_seeds_for_scatter=True):
    rows = read_csv(CANON / cycle / "per_seed.csv")
    by_dose = defaultdict(list)
    seeds_seen = sorted({r["seed"] for r in rows}, key=int)
    for r in rows:
        by_dose[r["dose"]].append(r)
    means, los, his = [], [], []
    for d in doses:
        vals = [float(r["test_ppl"]) for r in by_dose[d]]
        m, lo, hi = t_ci95(vals)
        means.append(m); los.append(m - lo); his.append(hi - m)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = list(range(len(doses)))
    st0 = series_style(0, palette)
    line_color = st0["color"] if palette == "color" else "0.05"
    ax.errorbar(x, means, yerr=[los, his], color=line_color, marker=st0["marker"],
                linestyle=st0["linestyle"], capsize=5, linewidth=2, markersize=7,
                label="mean [95% CI]", zorder=3)
    for si, s in enumerate(seeds_seen):
        yv = [float(next(r["test_ppl"] for r in by_dose[d] if r["seed"] == s)) for d in doses]
        gray = "0.6" if palette == "grayscale" else "grey"
        ax.plot(x, yv, marker=".", color=gray, alpha=0.6, markersize=5, linewidth=0.9,
                 zorder=2, label="per-seed" if si == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(doses)
    ax.set_ylabel("Held-out test PPL")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def make_fig05(palette):
    return make_dose_response(palette, "R013",
                               ["D_below", "D_xavier", "D_mid1", "D_mid2", "D_ctor"],
                               "R013 held-out test PPL over the five-dose target-RMS ladder")


def make_fig06(palette):
    return make_dose_response(palette, "R014",
                               ["D_xavier", "D_mid1", "D_ctor"],
                               "R014 held-out test PPL for the three tested RMS doses")


# =====================================================================
# fig07a / fig07b — R013 embedding RMS / gradient-L2 trajectories
# =====================================================================
def make_trajectory_fig(palette, value_col, ylabel, title):
    rows = read_csv(SRC / "dynamics_by_epoch_R013.csv")
    doses = ["D_below", "D_xavier", "D_mid1", "D_mid2", "D_ctor"]
    by_dose_epoch = defaultdict(list)
    for r in rows:
        by_dose_epoch[(r["dose"], int(r["epoch"]))].append(float(r[value_col]))

    fig, ax = plt.subplots(figsize=(8, 5.4))
    for i, d in enumerate(doses):
        st = series_style(i, palette)
        epochs = sorted({ep for (dose, ep) in by_dose_epoch if dose == d})
        ys = [stats.mean(by_dose_epoch[(d, ep)]) for ep in epochs]
        ax.plot(epochs, ys, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                 markersize=5, linewidth=1.8, label=d)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()
    return fig


def make_fig07a(palette):
    return make_trajectory_fig(palette, "embedding_rms_epoch_end",
                                "embedding RMS (epoch end, mean across seeds)",
                                "R013 token-embedding RMS trajectories")


def make_fig07b(palette):
    return make_trajectory_fig(palette, "embedding_grad_l2_mean",
                                "embedding grad L2 (mean across seeds)",
                                "R013 token-embedding gradient-L2 trajectories")


# =====================================================================
# fig07c — R014 best epoch by dose, per seed
# =====================================================================
def make_fig07c(palette):
    rows = read_csv(CANON / "R014" / "per_seed.csv")
    doses = ["D_xavier", "D_mid1", "D_ctor"]
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["dose"]] = int(r["best_epoch"])
    seeds = sorted(by_seed.keys(), key=int)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = list(range(len(doses)))
    for i, s in enumerate(seeds):
        st = series_style(i, palette)
        yv = [by_seed[s][d] for d in doses]
        ax.plot(x, yv, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                 markersize=7, linewidth=1.6, alpha=0.9, label=f"seed{s}")
    ax.set_xticks(x); ax.set_xticklabels(doses)
    ax.set_ylabel("selected best epoch (of 15)")
    ax.set_title("R014 best epoch by dose, per seed")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


# =====================================================================
# fig07d — R014 final vs best validation PPL
# =====================================================================
def make_fig07d(palette):
    rows = read_csv(CANON / "R014" / "per_seed.csv")
    doses = ["D_xavier", "D_mid1", "D_ctor"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, d in enumerate(doses):
        st = series_style(i, palette)
        bv = [float(r["best_val_ppl"]) for r in rows if r["dose"] == d]
        fv = [float(r["final_val_ppl"]) for r in rows if r["dose"] == d]
        ax.scatter(bv, fv, color=st["color"], marker=st["marker"], s=70, label=d,
                    edgecolor="black", linewidth=0.6)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.2)
    ax.set_xlabel("best val PPL")
    ax.set_ylabel("final (epoch 15) val PPL")
    ax.set_title("R014 final vs best validation PPL")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


FIGURES = [
    ("fig01_causal_reconstruction.png", make_fig01),
    ("fig02_R010_test_ppl.png", make_fig02),
    ("fig03_R011_factorial_test_ppl.png", make_fig03),
    ("fig04_R012_factorial_test_ppl.png", make_fig04),
    ("fig05_R013_dose_response_test_ppl.png", make_fig05),
    ("fig06_R014_dose_response_test_ppl.png", make_fig06),
    ("fig07a_R013_embedding_rms_trajectory.png", make_fig07a),
    ("fig07b_R013_embedding_grad_l2_trajectory.png", make_fig07b),
    ("fig07c_R014_best_epoch_by_dose.png", make_fig07c),
    ("fig07d_R014_final_vs_best_val_ppl.png", make_fig07d),
]

if __name__ == "__main__":
    for name, fn in FIGURES:
        for palette, out in (("color", OUT_COLOR), ("grayscale", OUT_GRAY)):
            fig = fn(palette)
            fig.savefig(out / name, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"{palette:10s} {name}")
    print("done:", len(FIGURES), "figures x 2 palettes =", len(FIGURES) * 2, "files")
