# Claims Ledger: Evidence Mapping

This document tracks the scientific claims allowed by the experimental data and explicitly forbids claims that are not supported.

## 1. Supported Claims

| Claim ID | Claim | Status | Evidence | Figure/Table | Risk |
|---|---|---|---|---|---|
| **C1** | Selective attention init improves PPL | Supported | R008b, R008-long, R009 | Fig 01, Fig 06, Tables 1-3 | Tested primarily on WikiText-2. |
| **C2** | Effect is not CE-specific | Supported | R008b (Orth/Gain > CE) | Fig 01, Table 1 | Deterministic vs stochastic trade-off exists. |
| **C3** | Spectral/Gain conditioning is the driver | Supported | R008b (Gain sweep) | Fig 02, Fig 03 | High sensitivity to specific gain (1.2). |
| **C4** | CE provides lower overfitting | Supported | R008-long, R009 | Fig 05, Fig 08 | Requires further dropout interaction study. |
| **C5** | Scale-up (4-layer) preserves effect | Supported | R009 | Fig 06, Fig 08 | Improvement magnitude decreases with scale. |

---

## 2. Prohibited Claims (Forbidden)

The following claims **MUST NOT** be made as they are not supported or are contradicted by the evidence:

- CE is uniquely superior: R008b clearly shows that Orthogonal and Gain-tuned Xavier perform better or equal in best PPL.
- The method is proven for large LLMs: Experiments were limited to Tiny Transformers (~2M to ~8M parameters).
- The method is SOTA: No comparison against heavy-regularized or industrial-scale baselines was performed.
- Operator Field Learning theory is fully validated: The theory provides a useful framework but the experiments only validate the initialization aspect, not the full learning dynamics theory.

---

## 3. Evidence Matrix

| Experiment | Key Metric | Xavier | Orthogonal | CE (LCG) | Delta (Best) |
|------------|------------|--------|------------|----------|--------------|
| R008b (2L) | Best PPL | 306.2 | 231.0 | 234.6 | -24.6% |
| R008-long (2L)| Best PPL | 308.4 | 227.9 | 230.0 | -26.1% |
| R009 (4L) | Best PPL | 219.6 | 189.1 | 202.5 | -13.9% |
