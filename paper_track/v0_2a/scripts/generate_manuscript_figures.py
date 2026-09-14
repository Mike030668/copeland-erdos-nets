#!/usr/bin/env python3
"""Generate the manuscript-created figures from canonical snapshots.
Canonical plots copied from accepted R011–R014 packages remain checksum-bound rather than regenerated here.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

steps = [
    "Historical selective-init gain",
    "R010\nattention-only attribution fails",
    "R011\nbenefit localizes to token embedding",
    "R012\ninitial RMS scale dominates redraw",
    "R013\nfresh-seed dose response",
    "R014\ntransfer to tested larger capacity",
]
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.axis("off")
for i, text in enumerate(steps):
    ax.text(i, 0.5, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5"))
    if i < len(steps)-1:
        ax.annotate("", xy=(i+0.72,0.5), xytext=(i+0.28,0.5), arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-0.6, len(steps)-0.4); ax.set_ylim(0,1); fig.tight_layout()
fig.savefig(OUT / "fig01_causal_reconstruction.png", dpi=200, bbox_inches="tight")
plt.close(fig)

r010 = pd.read_csv(ROOT / "canonical_inputs" / "R010" / "per_seed.csv")
order = ["xavier_g1.0","orthogonal","xavier_g1.2","ce_lcg"]
means = r010.groupby("method")["test_ppl"].mean().reindex(order)
fig, ax = plt.subplots(figsize=(7.2,4.8))
ax.bar(range(len(order)), means.values)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("Held-out test PPL"); ax.set_title("R010 corrected attention-only reconstruction")
fig.tight_layout(); fig.savefig(OUT / "fig02_R010_test_ppl.png", dpi=200, bbox_inches="tight"); plt.close(fig)
