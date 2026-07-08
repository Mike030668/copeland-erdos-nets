# Documentation

## Method

- **CE-LCG Initializer** — Copeland–Erdős digit stream → inverse normal CDF →
  LCG permutation → empirical standardization → scale by target σ.
  See `src/copeland_erdos_nets/ce_init.py` and `src/copeland_erdos_nets/assignment.py`
  for implementation details.

## Key Design Decisions

- **Layer-selective init**: Only attention projections (W_Q, W_K, W_V, W_O)
  receive custom initialization; FFN and embeddings use Xavier.
- **Matched controls**: Zero-mean CE-std-matched Gaussian/Uniform, not
  mean-matched. This isolates the spectral/structure effect from the scale effect.
- **LCG assignment**: Uses next power-of-two modulus M ≥ n with cycle walking,
  not a simple `π(j) = (aj + c) mod n`.
- **Empirical standardization**: CE-N explicitly subtracts the empirical mean
  and divides by empirical std before scaling: `W = σ · (z − μ) / s`.

## Reproducibility

See [`paper_artifacts/reproducibility.md`](../paper_artifacts/reproducibility.md)
for full environment setup and run commands.
