# Paper Artifacts

Canonical results, figures, and supporting documents for the article:

> **Layer-Selective Spectral Initialization of Attention Projections
> Improves Tiny Transformer Training**

## Directory Structure

```
paper_artifacts/
├── claims_ledger.md           — evidence-to-claim mapping + prohibited claims
├── reproducibility.md         — environment and run instructions
├── artifacts_checksums.txt    — SHA-256 checksums for all artifacts
├── tables/                    — canonical CSV result tables
│   ├── canonical_R008_selective_init.csv
│   ├── canonical_R008b_controls.csv
│   ├── canonical_R008_long.csv
│   ├── canonical_R009_scaleup.csv
│   ├── all_experiments_summary.csv
│   ├── seed_level_*.csv       — per-seed breakdowns
│   └── spectral_diagnostics_R008b.csv
├── figures/                   — publication figures (fig01–fig08)
├── raw/                       — original JSON result files from experiments
└── configs/                   — frozen config snapshots used in canonical runs
```

## Key Documents

- [Claims Ledger](claims_ledger.md) — what the data supports and what it does not.
- [Reproducibility Guide](reproducibility.md) — how to reproduce canonical runs.

## Figures

| Figure | Experiment | Content |
|---|---|---|
| fig01 | R008b | Best PPL comparison (bar chart) |
| fig02 | R008b | Gain sweep (Xavier gain = 1.0–1.6) |
| fig03 | R008b | Spectral conditioning (singular value distribution) |
| fig04 | R008-long | Weight-decay sweep results |
| fig05 | R008-long | Overfitting dynamics (train vs val PPL) |
| fig06 | R009 | 4-layer scale-up best PPL |
| fig07 | R009 | Learning curves |
| fig08 | R009 | Scale comparison (2-layer vs 4-layer) |
