# Experiment Configurations

## Transformer experiments (paper track)

| Config | Experiment | Role |
|---|---|---|
| `r008_selective_transformer.json` | R008 | Discovery: selective init of attention projections |
| `r008b_controls.json` | R008b | Main controlled comparison (8 methods, 5 seeds) |
| `r008_long_phase1.json` | R008-long | 50-epoch baseline / long training phase |
| `r008_long_phase2.json` | R008-long | Weight-decay sweep (WD = 0.01, 0.1, 0.5) |
| `r009_scaleup.json` | R009 | 4-layer scale-up validation (d=256) |
| `r007_transformer_screening.json` | R007 | Initial Transformer screening (exploratory) |

## Early experiments (MNIST / MLP-Mixer)

| Config | Experiment | Role |
|---|---|---|
| `mnist_screening.json` | R001 | Initial MNIST screening |
| `mnist_final_screening.json` | R003 | Final MNIST screening |
| `mnist_deep_mlp.json` | R004 | Deep MLP ablation |
| `mnist_deep_mlp_val.json` | R005 | Spectral diagnostics |
| `r005b_confirmation.json` | R005b | Spectral confirmation (shuffled assignment) |
| `r006b_mixer_screening.json` | R006 | MLP-Mixer screening |
| `r004_deep_mlp.json` | R004 | Deep MLP extended |
| `r005_diagnostics.json` | R005 | Init diagnostics config |
| `r005_screening_final.json` | R005 | Final screening with diagnostics |

## Ablation configs

| Config | Purpose |
|---|---|
| `ce_u_ablation.json` | CE-U mode ablation |
| `mnist_cnn_val.json` | CNN validation |
| `mnist_m_ablation.json` | Block width *m* ablation |
| `mnist_offset_sobol.json` | Sobol offset ablation |
| `sobol_u_ablation.json` | Sobol-U mode ablation |
| `mlp4_extended.json` | Extended MLP-4 config |

> **Note:** All Transformer configs use `vocab_size: 50257` (GPT-2 tokenizer).
> The `vocab_size` field in configs is informational; the runner loads the
> actual vocab size from `AutoTokenizer.from_pretrained("gpt2")` at runtime.
