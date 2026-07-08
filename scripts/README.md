# Reproducible Experiment Scripts

| Script | Purpose |
|---|---|
| `run_transformer_screening.py` | R007/R008/R008b/R008-long/R009 Transformer experiments on WikiText-2 |
| `run_mnist_screening.py` | Early MNIST screening (R001–R005b) |
| `run_mixer_screening.py` | MLP-Mixer screening (R006) |
| `collect_init_diagnostics.py` | Initialization / spectral diagnostics (SVD, cond number, autocorrelation) |
| `plot_mnist_results.py` | Plot MNIST screening results |
| `summarize_results.py` | Aggregate raw JSON results into summary tables |

## Usage

All experiment scripts accept `--config` and `--output` flags:

```bash
python scripts/run_transformer_screening.py \
  --config configs/r008b_controls.json \
  --output runs/r008b_controls
```
