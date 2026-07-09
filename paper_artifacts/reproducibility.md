# Reproducibility Guide

## 1. Environment

- **Python**: 3.10+
- **Hardware**: Nvidia T4 (16 GB) or better. CPU-only works but is very slow.
- **Key dependencies** (installed via `make install`):
  - torch >= 2.0
  - transformers >= 4.30
  - datasets >= 2.0
  - numpy, scipy, matplotlib, sympy

## 2. Installation

```bash
git clone https://github.com/Mike030668/copeland-erdos-nets.git
cd copeland-erdos-nets
make install
make check-env
```

## 3. Tokenizer / Vocab Size

All Transformer experiments use the HuggingFace GPT-2 tokenizer
(`AutoTokenizer.from_pretrained("gpt2")`), which has **vocab_size = 50 257**.

The `vocab_size` field in config JSON files is informational and reflects this
value. The runner loads the actual vocab size from the tokenizer at runtime.

## 4. Running Canonical Experiments

### R008b — Controlled comparison (2-layer, 15 epochs)

```bash
python scripts/run_transformer_screening.py \
  --config configs/r008b_controls.json \
  --output runs/r008b_controls
```

### R008-long — 50-epoch + weight-decay sweep

```bash
python scripts/run_transformer_screening.py \
  --config configs/r008_long_phase1.json \
  --output runs/r008_long_phase1

python scripts/run_transformer_screening.py \
  --config configs/r008_long_phase2.json \
  --output runs/r008_long_phase2
```

### R009 — 4-layer scale-up (25 epochs)

```bash
python scripts/run_transformer_screening.py \
  --config configs/r009_scaleup.json \
  --output runs/r009_scaleup
```

## 5. Raw Data and Checksums

Canonical raw JSON results are in `paper_artifacts/raw/`.
SHA-256 checksums for all artifacts are in `paper_artifacts/artifacts_checksums.txt`.

## 6. Canonical Tables

| File | Experiment |
|---|---|
| `tables/canonical_R008_selective_init.csv` | R008 (discovery) |
| `tables/canonical_R008b_controls.csv` | R008b (controlled comparison) |
| `tables/canonical_R008_long.csv` | R008-long (50-epoch) |
| `tables/canonical_R009_scaleup.csv` | R009 (4-layer scale-up) |
| `tables/all_experiments_summary.csv` | All experiments combined |
| `tables/seed_level_*.csv` | Per-seed breakdowns |
| `tables/spectral_diagnostics_R008b.csv` | Spectral diagnostics from R008b |
